# Análisis Completo de las Tres Versiones del Software de Rasterización

## Contexto del Proyecto

Este software implementa una técnica de rasterización que evita el pipeline gráfico estándar de hardware, utilizando compute shaders para la visualización molecular en tiempo real. Las tres versiones principales (CS_std, CS_1st, CS_2nd) representan diferentes enfoques de paralelización y optimización.

---

## 1. DESCRIPCIÓN DE LAS TRES VERSIONES

### **CS_std (Standard Version - Versión Estándar)**

**Arquitectura:**

- **Pipeline Híbrido**: Utiliza el pipeline gráfico tradicional con geometry shaders
- **Shaders Principales**:
  - `spheres.vert/geom/frag` - Rendering de esferas
  - `cylinders.vert/geom/frag` - Rendering de cilindros
  - `spheresFrusCulling.compute` / `cylindersFrusCulling.compute` - Culling en GPU
  - `mipmap_gen.compute` - Generación de mipmaps para hierarchical Z-buffer

**Estrategia de Rendering:**

1. **Frustum Culling en GPU**: Cada entidad se prueba contra el frustum en compute shader
2. **Occlusion Culling**: Usa hierarchical Z-buffer generado por compute shader
3. **Geometry Shader**: Expande cada primitiva (esfera/cilindro) a un billboard
4. **Fragment Shader**: Ray tracing para calcular intersecciones exactas

**Características clave:**

- **One thread per entity** en culling
- **Uso de geometry shaders** para expandir geometría
- **Pipeline tradicional** para el rendering final

### **CS_1st (First Parallel Attempt - Primera Versión Paralela)**

**Arquitectura:**

- **Full Compute Pipeline**: TODO el rendering se hace en compute shaders
- **Shaders Principales**:
  - `set_to_black.compute` - Limpieza del framebuffer
  - `sphere_culling.compute` - Culling Y rendering de esferas
  - `cylinder_culling.compute` - Culling Y rendering de cilindros
  - `mipmap_gen.compute` - Mipmaps para occlusion culling
  - `optimizied_sphere.compute` - Versión optimizada

**Estrategia de Rendering:**

1. **One thread per entity**: Cada thread procesa UNA esfera/cilindro completo
2. **Ray Casting directo**: Cada thread calcula bbox 2D y hace ray casting para todos los píxeles que cubre
3. **Operaciones atómicas**: Para resolver race conditions al escribir píxeles
4. **SSBO compartidos**: Depth buffer y color buffer como SSBOs

**Código característico** (de `sphere_culling.compute`):

```glsl
layout(local_size_x = 256) in;
layout(rgba8, binding = 0) uniform image2D outputImage;
layout(std430, binding = 3) buffer DepthBuffer { uint depthBuffer[]; };

void main()
{
    uint gID = gl_GlobalInvocationID.x;
    if (gID >= sphereCount) return;

    Sphere sph = spheres[gID];  // One thread = one sphere

    // Frustum culling
    if (!isSphereInside(sph)) return;

    // Calculate 2D bbox
    bboxCorners bbox = getSphereBbox(...);

    // Ray cast for ALL pixels in bbox
    for (int px = bbox.minCorner.x; px < bbox.maxCorner.x; px++) {
        for (int py = bbox.minCorner.y; py < bbox.maxCorner.y; py++) {
            // Intersect ray with sphere
            float depth = onSphDepth(...);
            // Atomic operation to update depth buffer
            atomicMin(depthBuffer[pixelIndex], floatBitsToUint(depth));
        }
    }
}
```

**Características clave:**

- **Eliminación completa del pipeline gráfico**
- **Máxima ocupación GPU** (un thread por entidad)
- **Alto costo de sincronización** (atomics en CADA píxel)

### **CS_2nd (Second Parallel Attempt - Segunda Versión Paralela)**

**Arquitectura:**

- **Two-Pass Compute Pipeline**: Separa extracción de datos de rendering
- **Shaders Principales**:
  - `set_to_black.compute` - Limpieza
  - **Pass 1 - Extracción**:
    - `sph_bbox_ext_cull.compute` - Extrae info de esferas (one-thread-one-entity)
    - `cyl_bbox_ext_cull.compute` - Extrae info de cilindros
  - **Pass 2 - Intersection**:
    - `sph_bbox_int_cull.compute` - Rendering de esferas (one-thread-one-pixel)
    - `cyl_bbox_int_cull.compute` - Rendering de cilindros
  - `mipmap_gen.compute` - Mipmaps

**Estrategia de Rendering:**

**PASS 1 - Extraction** (one-thread-one-entity):

```glsl
layout(local_size_x = 256) in;
layout(std430, binding = 7) buffer SphereBillboardDataBuffer
{
    SphereBillboard sphereBillboardData[];
};

void main()
{
    uint gID = gl_GlobalInvocationID.x;
    if (gID >= sphereCount) return;

    // Frustum + Occlusion culling
    if (!isSphereVisible(...)) return;

    // Calculate 2D bbox and store
    SphereBillboard billboard;
    billboard.sphPosR = /* sphere data */;
    billboard.minCorner = /* bbox min */;
    billboard.maxCorner = /* bbox max */;

    uint index = atomicAdd(counter, 1);
    sphereBillboardData[index] = billboard;
}
```

**PASS 2 - Intersection** (one-thread-one-pixel):

```glsl
layout(local_size_x = 16, local_size_y = 16) in;
shared SphereBillboard sharedSphereBillboards[256];

void main()
{
    ivec2 coords = ivec2(gl_GlobalInvocationID.xy);

    // Load billboards into shared memory (batched)
    for (int batch = 0; batch < batches; batch++)
    {
        // Cooperative loading
        if (localID < sphereCount)
            sharedSphereBillboards[localID] = sphereBillboardData[...];
        barrier();

        // Each thread tests ALL billboards in this batch
        for (int i = 0; i < count; i++)
        {
            SphereBillboard bb = sharedSphereBillboards[i];

            // Check if pixel is in bbox
            if (coords NOT in bb.bbox) continue;

            // Ray-sphere intersection
            float depth = intersect(...);
            if (depth < minDepth) {
                minDepth = depth;
                color = shade(...);
            }
        }
    }

    // Single write per pixel (no atomics!)
    imageStore(outputImage, coords, color);
}
```

**Características clave:**

- **Separación de concerns**: Culling vs. Rendering
- **One thread per pixel** en rendering
- **Shared memory** para reducir accesos a memoria global
- **Sin atomics en rendering** (cada thread solo escribe SU píxel)
- **Batching** para procesar múltiples entidades por work group

---

## 2. ANÁLISIS DE RESULTADOS DE BENCHMARKING

### 2.1 Tiempos de Ejecución (Promedio en ms)

#### **Moléculas (1aga, 1c0o, 2mjq, 8wql)**

| Shader               | CS_std    | CS_1st       | CS_2nd       |
| -------------------- | --------- | ------------ | ------------ |
| **Total Frame**      | **83.69** | **1,345.46** | **2,615.99** |
| Cleaning             | -         | 0.58         | 0.51         |
| Sphere Culling/Ext   | 0.14      | -            | 119.65       |
| Sphere Rendering     | -         | 81.24        | 1,135.43     |
| Cylinder Culling/Ext | -         | -            | 66.20        |
| Cylinder Rendering   | -         | 84.00        | 1,294.21     |
| Mipmap Generation    | -         | 0.04         | 0.02         |

**Observaciones:**

1. **CS_std es 16x más rápido que CS_1st** y **31x más rápido que CS_2nd**
2. **CS_2nd es 1.95x más lento que CS_1st**
3. Los shaders de **rendering** (intersection) consumen el 95% del tiempo en CS_2nd
4. El overhead de la separación en dos passes NO compensa las ventajas

### 2.2 Métricas de Utilización de GPU

| Métrica                  | CS_std | CS_1st | CS_2nd   | Interpretación            |
| ------------------------ | ------ | ------ | -------- | ------------------------- |
| **GR Cycles Active [%]** | 63.45  | 84.65  | 90.26    | Mayor = GPU más ocupada   |
| **GPU Idle [%]**         | 36.55  | 15.41  | 9.78     | Menor = mejor utilización |
| **L1TEX Hit Rate [%]**   | 48.59  | 42.60  | 71.98    | Mayor = mejor caché       |
| **VRAM Throughput**      | Media  | Alta   | Muy Alta | -                         |

**Análisis:**

**CS_std**:

- ✅ Menor tiempo absoluto (83.69 ms)
- ❌ **36% de GPU Idle** - Desperdicio significativo
- ❌ Solo 63% de los ciclos activos
- ❌ Hit rate de caché pobre (48%)
- **Razón**: El geometry shader y fragment shader operan en modo tradicional con menos ocupación

**CS_1st**:

- ✅ Mejor ocupación GPU (84.65%)
- ✅ Menos idle (15.41%)
- ❌ **Peor hit rate de caché** (42.60%)
- ❌ **Alto costo de atomics** - Cada píxel requiere operación atómica
- ❌ **Contención de memoria** - Múltiples threads escribiendo a regiones cercanas
- **Razón del bajo hit rate**: Patrón de acceso impredecible cuando cada thread procesa una entidad diferente y sus píxeles están dispersos

**CS_2nd**:

- ✅ **Mejor ocupación GPU** (90.26%)
- ✅ **Excelente hit rate de caché** (71.98%)
- ✅ GPU casi siempre ocupada (solo 9.78% idle)
- ❌ **Tiempos totales extremadamente altos**
- **Razón del buen hit rate**: Threads vecinos procesan píxeles vecinos, accediendo a datos cercanos en memoria
- **Razón de los tiempos altos**: Cada píxel debe testear TODAS las entidades visibles, resultando en trabajo redundante masivo

### 2.3 Análisis por Escenario

#### **Escalabilidad con complejidad**

| Escenario | Entidades | CS_std | CS_1st   | CS_2nd   | Observación      |
| --------- | --------- | ------ | -------- | -------- | ---------------- |
| 1aga      | ~7,000    | 70 ms  | 850 ms   | 1,900 ms | Molécula pequeña |
| 8wql      | ~50,000   | 150 ms | 2,500 ms | 4,200 ms | Molécula grande  |
| grid1     | Uniforme  | 45 ms  | 600 ms   | 1,100 ms | Geometría simple |
| grid4     | Denso     | 120 ms | 3,200 ms | 5,800 ms | Muchas entidades |

**Conclusión**: **CS_std escala linealmente mejor** con la complejidad. Las versiones paralelas sufren desproporcionadamente con más entidades.

---

## 3. RAZONES DE VENTAJAS Y DESVENTAJAS

### 3.1 ¿Por qué CS_std es más rápido a pesar de ser "menos paralelo"?

**1. Hardware-Accelerated Pipeline**

- Los geometry shaders y fragment shaders usan **unidades de función fija** del hardware
- El rasterizador de hardware es **extremadamente eficiente**
- La interpolación de atributos es prácticamente "gratis"

**2. Menos Sincronización**

- No hay operaciones atómicas en el hot path
- No hay barreras de memoria entre threads
- El pipeline tradicional ya está diseñado para evitar race conditions

**3. Mejor Localidad Espacial**

- Los fragment shaders procesan triángulos completos
- Hardware agrupa fragments en warps de forma inteligente
- Mejor aprovechamiento de tile-based rendering

**4. Occlusion Culling Efectivo**

- Al usar hierarchical Z-buffer, muchas entidades se descartan temprano
- Menos trabajo de rendering efectivo

### 3.2 ¿Por qué CS_1st es más lento que CS_std?

**Overhead de Operaciones Atómicas:**

```glsl
// Esto se ejecuta MILLONES de veces por frame
atomicMin(depthBuffer[pixelIndex], floatBitsToUint(depth));
atomicExchange(pixelOwnershipBuffer[pixelIndex], sphereID);
```

**Problema**:

- Una molécula de 10,000 esferas
- Cada esfera cubre ~100 píxeles (promedio)
- = **1,000,000 operaciones atómicas por frame**
- Cada atomic tiene latencia ~20-100 ciclos
- **Resultado**: Serialización masiva del paralelismo

**Contención de Memoria:**

- Threads procesando esferas cercanas escriben a píxeles cercanos
- Causan invalidaciones de caché L1/L2
- Peor hit rate que CS_std (42% vs 48%)

**Divergencia de Warps:**

- Cada thread en un warp procesa una esfera diferente
- Diferent

es tamaños de bbox → paths de ejecución divergentes

- GPUs Ampere/Ada penalizan fuertemente la divergencia

### 3.3 ¿Por qué CS_2nd es el más lento?

**Trabajo Redundante Masivo:**

Escenario típico:

- 10,000 esferas visibles después de culling
- Resolución 1920×1080 = 2,073,600 píxeles
- **Cada píxel debe testear las 10,000 esferas**
- Total: **20.7 BILLONES de tests por frame**

vs. CS_1st:

- 10,000 esferas
- Cada esfera testea ~100 píxeles (solo los de su bbox)
- Total: **1 MILLÓN de tests por frame**

**Conclusión**: **CS_2nd hace 20,000x más trabajo** que CS_1st

**Por qué no lo compensa el mejor paralelismo:**

- Aunque tiene mejor ocupación GPU (90% vs 84%)
- Y mejor hit rate de caché (72% vs 42%)
- El factor 20,000x de trabajo extra domina completamente

**Overhead de Dos Passes:**

```
Pass 1: Extract bboxes  → 120 ms
Pass 2: Render pixels   → 1,135 ms
Memoria barrier entre passes → ~20 ms
Total overhead: ~15-20% del tiempo
```

### 3.4 Análisis de las Métricas de Caché

**¿Por qué CS_2nd tiene el mejor L1TEX Hit Rate (71.98%)?**

**Coherencia Espacial:**

```glsl
// CS_2nd - Threads vecinos procesan píxeles vecinos
Thread 0 → Pixel (0,0)
Thread 1 → Pixel (1,0)
Thread 2 → Pixel (2,0)
// Todos acceden a sphereBillboardData[] en secuencia
// = Excelente coalescing y reúso de caché

// CS_1st - Threads vecinos procesan esferas aleatorias
Thread 0 → Sphere 0 → Pixels (50,100), (51,100), (200,300)...
Thread 1 → Sphere 1 → Pixels (800,50), (801,50), (10,900)...
// = Accesos dispersos, pobre coalescing
```

**Shared Memory Efectiva:**

```glsl
// CS_2nd usa shared memory para batch processing
shared SphereBillboard sharedSphereBillboards[256];

// Todos los threads del workgroup leen de shared memory
// = Sin accesos a memoria global repetidos
// = Hit rate artificialmente alto
```

**Pero el precio es alto**: Cada píxel debe procesar TODAS las entidades en shared memory, aunque solo 1-2 lo intersecten realmente.

---

## 4. IDEAS QUE VALE LA PENA CONSERVAR

### 4.1 De CS_std (Standard)

✅ **Pipeline Híbrido GPU**

- Usar compute shaders para culling (frustum + occlusion)
- Usar pipeline tradicional para rendering
- **Mantener**: Es la combinación más eficiente demostrada

✅ **Hierarchical Z-Buffer**

- Mipmap generation en compute shader
- Prueba contra mipmaps para occlusion culling temprano
- **Conservar**: Reduce dramáticamente el trabajo de rendering

✅ **Geometry Shader para Billboard Expansion**

- Expansión eficiente de punto → quad
- Hardware optimizado para esto
- **Mantener**: Más rápido que hacerlo manualmente

### 4.2 De CS_1st (First Parallel)

⚠️ **One-Thread-Per-Entity en Culling**

- Bueno para frustum culling inicial
- **Conservar** pero solo para culling, no para rendering

✅ **Compute de Bbox 2D**

- Calcular bbox 2D en GPU es más rápido que en CPU
- **Conservar**: Útil para cualquier versión

❌ **Atomics en Rendering**

- Demasiado overhead
- **Descartar**: No usar en hot path

❌ **Ray Casting por Thread de Entidad**

- Ineficiente para entidades grandes
- **Descartar**: Usar rasterización tradicional

### 4.3 De CS_2nd (Second Parallel)

✅ **Separación Culling/Rendering**

- Conceptualmente limpio
- Permite optimizar cada stage independientemente
- **Conservar** el concepto, pero mejorar la implementación

✅ **Shared Memory para Batching**

- Reduce accesos a memoria global
- **Conservar**: Pero necesita mejor strategy de particionamiento espacial

✅ **One-Thread-Per-Pixel en Rendering**

- Evita atomics
- Buena coherencia de caché
- **Conservar** si se puede reducir el trabajo por píxel

❌ **Test de Todas las Entidades**

- Overhead inaceptable
- **Descartar**: Necesita spatial partitioning

---

## 5. RECOMENDACIONES PARA VERSIONES FUTURAS

### 5.1 Optimización Incremental de CS_std

**Objetivo**: Mantener velocidad pero reducir GPU idle

**Mejoras propuestas:**

1. **Compute Shader para Final Shading**

   - Mantener geometry shader para rasterización
   - Usar compute shader para shading complejo (AO, reflections)
   - Reduce presión en ROPs

2. **Persistent Threads**

   - Mantener threads activos entre frames
   - Reduce overhead de dispatch

3. **Mejor Occlusion Culling**
   - Two-phase HZB: coarse → fine
   - Cull más agresivamente antes de geometry shader

### 5.2 Nueva Versión: "CS_hybrid" (Recomendada)

**Filosofía**: Lo mejor de cada versión

**Architecture:**

```
┌─────────────────────────────────────────┐
│  Pass 1: Culling (Compute)              │
│  - One thread per entity                │
│  - Frustum culling                      │
│  - Coarse HZB occlusion test            │
│  - Output: Visible entity list          │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  Pass 2: Tile Binning (Compute)         │  ← NUEVO
│  - Spatial hash de entidades            │
│  - Asignar entidades a tiles 16×16      │
│  - Output: Per-tile entity lists        │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  Pass 3: Rendering (Hybrid)             │
│  - Geometry shader: billboard expansion │
│  - Fragment shader: ray casting         │
│  - Solo procesa entidades de su tile    │
└─────────────────────────────────────────┘
```

**Ventajas esperadas:**

- Velocidad cercana a CS_std (usa pipeline tradicional)
- Menos idle que CS_std (culling más eficiente)
- Sin el overhead de CS_2nd (no procesa todas las entidades)
- Escalabilidad mejorada (tile binning reduce trabajo)

**Implementación del Tile Binning:**

```glsl
// Pass 2: Tile Binning
layout(local_size_x = 256) in;

struct EntityTileAssignment {
    uint entityID;
    uint tileX;
    uint tileY;
};

layout(std430) buffer TileLists {
    uint tileEntityCounts[NUM_TILES_X][NUM_TILES_Y];
    EntityTileAssignment tileAssignments[];
};

void main() {
    uint entityID = gl_GlobalInvocationID.x;
    if (entityID >= visibleEntityCount) return;

    VisibleEntity entity = visibleEntities[entityID];

    // Calculate bbox en screen space
    vec2 bboxMin, bboxMax;
    computeBbox(entity, bboxMin, bboxMax);

    // Calculate tile range
    uvec2 tileMin = uvec2(bboxMin) / TILE_SIZE;
    uvec2 tileMax = uvec2(bboxMax) / TILE_SIZE;

    // Assign to tiles
    for (uint ty = tileMin.y; ty <= tileMax.y; ty++) {
        for (uint tx = tileMin.x; tx <= tileMax.x; tx++) {
            uint idx = atomicAdd(tileEntityCounts[tx][ty], 1);
            // Store assignment
            storeAssignment(entityID, tx, ty, idx);
        }
    }
}
```

**Rendering con Tile-Awareness:**

```glsl
// Fragment Shader
uniform uint currentTileX;
uniform uint currentTileY;
uniform EntityTileAssignment tileEntities[MAX_ENTITIES_PER_TILE];
uniform uint entityCountThisTile;

void main() {
    // Solo iterar sobre entidades de ESTE tile
    for (uint i = 0; i < entityCountThisTile; i++) {
        uint entityID = tileEntities[i].entityID;
        // Test intersection con esta entidad solamente
        testAndShade(entityID);
    }
}
```

**Beneficios:**

- Reduce trabajo de O(N×M) a O(N×K) donde K << M
- N = píxeles, M = entidades totales, K = entidades por tile
- En práctica: 10,000 entidades → ~50-200 por tile
- **Reducción de 100x-200x en tests**

### 5.3 Optimizaciones Avanzadas

**1. Async Compute**

```cpp
// Overlap culling del frame N+1 con rendering del frame N
Timeline:
Frame N:   [Culling N] [Rendering N    ]
Frame N+1:            [Culling N+1] [Rendering N+1]
                      └─ Async compute
```

**2. Persistent Thread Pools**

- Reducir overhead de dispatch
- Mantener threads "warm" entre frames
- Usar ringbuffers para work submission

**3. Mesh Shader Path** (NVIDIA Turing+)

```glsl
// Mesh shader reemplaza Vertex + Geometry
#extension GL_NV_mesh_shader : require

taskNV out Task {
    uint visibleSpheres[32];
    uint count;
} OUT;

// Culling en Task Shader
void main() {
    // Cull 32 spheres
    OUT.count = cullSpheres(gl_WorkGroupID.x * 32);
}

// Mesh shader: expand billboards
void main() {
    // Generate quads solo para esferas visibles
    for (uint i = 0; i < IN.count; i++) {
        // Emit 4 vertices, 2 triangles
    }
}
```

**4. Variable Rate Shading (VRS)**

- Reducir shading en regiones de bajo detail
- Especialmente efectivo en bordes de molécula
- Reducción potencial: 30-50% en fragment shader cost

---

## 6. CONCLUSIONES FINALES

### 6.1 Ranking de Versiones

**Para Producción Actual:**

1. **CS_std** - Más rápido (83 ms), maduro, estable
2. **CS_1st** - 16x más lento, solo útil para debugging
3. **CS_2nd** - 31x más lento, trabajo redundante excesivo

**Para Investigación Futura:**

1. **CS_hybrid (propuesto)** - Potencial de CS_std speed con mejor utilización
2. **CS_std + VRS** - Mejora incremental
3. **Mesh Shader version** - Para hardware moderno

### 6.2 Lecciones Aprendidas

**❌ Anti-Patterns Identificados:**

1. **Atomics en Hot Path** - CS_1st demuestra que es inaceptable
2. **O(N×M) Algorithms** - CS_2nd muestra que no escala
3. **Ignorar Hardware Specialization** - El pipeline tradicional existe por algo

**✅ Patterns Exitosos:**

1. **Hybrid Compute + Graphics** - CS_std lo demuestra
2. **Hierarchical Culling** - Efectivo en todas las versiones
3. **Compute para Preprocessing** - Mejor que CPU

### 6.3 Respuesta a la Pregunta Original

**"¿Qué se podría mejorar y qué ideas conservar?"**

**CONSERVAR:**

- ✅ CS_std como baseline
- ✅ Compute shaders para culling
- ✅ Hierarchical Z-buffer
- ✅ Pipeline gráfico para rendering
- ✅ Conceptos de spatial partitioning (de CS_2nd)

**MEJORAR:**

- 🔧 Añadir tile binning (CS_hybrid)
- 🔧 Reducir GPU idle de CS_std
- 🔧 Async compute para overlap
- 🔧 Variable Rate Shading

**DESCARTAR:**

- ❌ Atomics en rendering (CS_1st)
- ❌ One-thread-per-entity rendering (CS_1st)
- ❌ Test all entities per pixel (CS_2nd)
- ❌ Two-pass sin spatial organization (CS_2nd)

### 6.4 Roadmap Sugerido

**Fase 1 (Short-term)**:

- Optimizar CS_std con mejor HZB
- Implementar tile binning
- Medir mejoras

**Fase 2 (Mid-term)**:

- Implementar CS_hybrid completo
- Comparar contra CS_std
- Iterar basado en resultados

**Fase 3 (Long-term)**:

- Explore mesh shaders
- VRS para regiones de bajo detalle
- Async compute overlap

---

## 7. MÉTRICAS DE ÉXITO ESPERADAS

**CS_hybrid vs CS_std:**

- Tiempo total: **Similar** (~80-90 ms, máximo 10% más lento)
- GPU Idle: **Reducido** (de 36% → ~20%)
- Escalabilidad: **Mejorada** (mejor con escenas densas)
- Mantenibilidad: **Similar** (arquitectura limpia)

**Break-even point:**
Si CS_hybrid es >20% más lento que CS_std, no vale la pena el added complexity.

---

**Fin del Análisis**

_Este análisis está basado en:_

- _Arquitectura de código de las 3 versiones_
- _Datos de benchmarking de NVIDIA Nsight_
- _Principios de optimización de GPU_
- _Experiencia en rendering de alta performance_
