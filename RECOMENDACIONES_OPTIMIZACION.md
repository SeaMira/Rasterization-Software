# Recomendaciones de Optimización - Software de Rasterización Molecular

## Resumen Ejecutivo

Este documento proporciona recomendaciones específicas de optimización para las diferentes versiones del software de rasterización molecular (fst_parallel, snd_parallel, cuda, standard_version) basadas en el análisis del código y los benchmarks existentes.

**Estado Actual:**
- **standard_version**: La más rápida (83 ms/frame) pero con 36% GPU idle
- **fst_parallel**: 16x más lenta debido a operaciones atómicas masivas
- **snd_parallel**: 31x más lenta debido a trabajo redundante O(N×M)
- **cuda**: Similar a fst_parallel pero con potencial de optimización CUDA

---

## 1. OPTIMIZACIONES PARA STANDARD_VERSION (Prioridad Alta)

### 1.1 Reducir GPU Idle (36% → <20%)

**Problema Identificado:**
- El pipeline tradicional deja la GPU inactiva durante partes del frame
- El geometry shader no mantiene ocupación constante

**Soluciones:**

#### A. Async Compute Overlap
```cpp
// En el loop principal, solapar culling del frame N+1 con rendering del frame N
void renderFrame() {
    // Frame N
    if (asyncComputeQueue) {
        // Iniciar culling del frame N+1 en async compute queue
        dispatchAsyncCulling(frameNplus1);
    }
    
    // Renderizar frame N
    renderWithGraphicsPipeline(frameN);
    
    // Sincronizar antes de usar resultados
    syncAsyncCompute();
}
```

**Beneficio esperado:** Reducción de 10-15% en GPU idle

#### B. Persistent Threads para Culling
```glsl
// Mantener threads activos entre frames
layout(local_size_x = 256) in;

// Usar ring buffer para work submission
layout(std430, binding = 0) buffer WorkQueue {
    uint entityIndices[];
    atomic_uint queueHead;
    atomic_uint queueTail;
};

void main() {
    // Threads persistentes procesan work queue
    while (hasWork()) {
        uint entityID = dequeueWork();
        processEntity(entityID);
    }
}
```

**Beneficio esperado:** Reducción de overhead de dispatch, 5-8% mejora

### 1.2 Optimizar Hierarchical Z-Buffer

**Problema:** Generación de mipmaps puede ser más eficiente

**Solución: Two-Phase HZB**

```glsl
// Pass 1: Coarse culling con mipmaps bajos
layout(local_size_x = 256) in;

void main() {
    uint entityID = gl_GlobalInvocationID.x;
    
    // Test contra mipmap nivel 3 (muy bajo res)
    if (isOccludedCoarse(entityID, hizLevel3)) {
        return; // Early exit
    }
    
    // Solo entidades que pasan coarse test van a fine test
    uint fineIndex = atomicAdd(fineTestCounter, 1);
    fineTestEntities[fineIndex] = entityID;
}

// Pass 2: Fine culling solo para candidatos
void fineCulling() {
    // Test contra mipmap nivel 1 (alta res)
    // Muchas menos entidades a procesar
}
```

**Beneficio esperado:** 15-25% reducción en tiempo de culling

### 1.3 Variable Rate Shading (VRS)

**Aplicar VRS en regiones de bajo detalle:**

```cpp
// Configurar VRS basado en densidad de entidades
void setupVRS() {
    // Regiones con muchas entidades: shading rate 1x1
    // Regiones con pocas entidades: shading rate 2x2 o 4x4
    glShadingRateImageNV(vrsTexture);
}
```

**Beneficio esperado:** 20-30% reducción en fragment shader cost

### 1.4 Optimizar Geometry Shader

**Problema:** Geometry shader puede ser cuello de botella

**Solución: Mesh Shaders (NVIDIA Turing+)**

```glsl
#extension GL_NV_mesh_shader : require

// Task shader: culling y determinación de billboards
taskNV out Task {
    uint visibleSpheres[32];
    uint count;
} OUT;

void main() {
    // Cull 32 esferas por task
    OUT.count = cullSpheresBatch(gl_WorkGroupID.x * 32);
}

// Mesh shader: generar quads
void main() {
    // Generar billboards solo para esferas visibles
    for (uint i = 0; i < IN.count; i++) {
        emitBillboard(IN.visibleSpheres[i]);
    }
}
```

**Beneficio esperado:** 30-40% mejora si hardware lo soporta

---

## 2. OPTIMIZACIONES PARA FST_PARALLEL (Prioridad Media)

### 2.1 Eliminar Operaciones Atómicas en Hot Path

**Problema Crítico:** 1M+ operaciones atómicas por frame

**Solución: Tile-Based Rendering**

```glsl
// Dividir pantalla en tiles 16x16
const uint TILE_SIZE = 16;
const uint TILES_X = screenResolution.x / TILE_SIZE;
const uint TILES_Y = screenResolution.y / TILE_SIZE;

// Pass 1: Asignar entidades a tiles
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer TileLists {
    uint tileEntityCounts[TILES_X * TILES_Y];
    uint tileEntityIndices[]; // Compact list
};

void main() {
    uint entityID = gl_GlobalInvocationID.x;
    if (entityID >= sphereCount) return;
    
    Sphere s = spheres[entityID];
    bboxCorners bbox = getSphereBbox(s);
    
    // Calcular tiles que intersecta
    ivec2 tileMin = ivec2(bbox.minCorner) / TILE_SIZE;
    ivec2 tileMax = ivec2(bbox.maxCorner) / TILE_SIZE;
    
    // Asignar a tiles (una atomic por tile, no por pixel)
    for (int ty = tileMin.y; ty <= tileMax.y; ty++) {
        for (int tx = tileMin.x; tx <= tileMax.x; tx++) {
            uint tileIdx = ty * TILES_X + tx;
            uint idx = atomicAdd(tileEntityCounts[tileIdx], 1);
            tileEntityIndices[tileIdx * MAX_ENTITIES_PER_TILE + idx] = entityID;
        }
    }
}

// Pass 2: Renderizar por tile (sin atomics)
layout(local_size_x = 16, local_size_y = 16) in;

void main() {
    ivec2 tileCoord = ivec2(gl_WorkGroupID.xy);
    ivec2 pixelCoord = ivec2(gl_LocalInvocationID.xy) + tileCoord * TILE_SIZE;
    
    uint tileIdx = tileCoord.y * TILES_X + tileCoord.x;
    uint entityCount = tileEntityCounts[tileIdx];
    
    float minDepth = 1.0;
    vec4 finalColor = vec4(0.0);
    
    // Solo testear entidades de este tile
    for (uint i = 0; i < entityCount; i++) {
        uint entityID = tileEntityIndices[tileIdx * MAX_ENTITIES_PER_TILE + i];
        Sphere s = spheres[entityID];
        
        // Test intersection
        float depth = intersectSphere(pixelCoord, s);
        if (depth < minDepth) {
            minDepth = depth;
            finalColor = shadeSphere(s, depth);
        }
    }
    
    // Una escritura por pixel, sin atomics
    imageStore(outputImage, pixelCoord, finalColor);
}
```

**Beneficio esperado:** Reducción de 50-70% en tiempo de rendering

### 2.2 Optimizar Acceso a Memoria

**Problema:** Bajo hit rate de caché (42%)

**Solución: Pre-sort Entities por Screen Position**

```cpp
// CPU: Ordenar esferas por posición en pantalla antes de enviar a GPU
void sortSpheresByScreenPosition(std::vector<Sphere>& spheres, const Camera& camera) {
    std::sort(spheres.begin(), spheres.end(), [&](const Sphere& a, const Sphere& b) {
        vec3 aScreen = projectToScreen(a.position, camera);
        vec3 bScreen = projectToScreen(b.position, camera);
        
        // Ordenar por tile primero, luego por Y, luego por X
        ivec2 aTile = ivec2(aScreen.xy) / TILE_SIZE;
        ivec2 bTile = ivec2(bScreen.xy) / TILE_SIZE;
        
        if (aTile.y != bTile.y) return aTile.y < bTile.y;
        if (aTile.x != bTile.x) return aTile.x < bTile.x;
        return aScreen.y < bScreen.y;
    });
}
```

**Beneficio esperado:** Mejora de hit rate a 60-70%

### 2.3 Usar Shared Memory Efectivamente

```glsl
layout(local_size_x = 16, local_size_y = 16) in;

shared Sphere sharedSpheres[64]; // Cache de esferas del tile

void main() {
    uint localID = gl_LocalInvocationID.x + gl_LocalInvocationID.y * 16;
    ivec2 pixelCoord = ivec2(gl_GlobalInvocationID.xy);
    
    // Cargar esferas del tile en shared memory (cooperativo)
    uint tileEntityCount = tileEntityCounts[tileIdx];
    for (uint i = localID; i < tileEntityCount; i += 256) {
        if (i < 64) { // Solo las primeras 64 caben en shared
            sharedSpheres[i] = spheres[tileEntityIndices[tileIdx * MAX_ENTITIES + i]];
        }
    }
    barrier();
    
    // Procesar desde shared memory
    uint sharedCount = min(64u, tileEntityCount);
    for (uint i = 0; i < sharedCount; i++) {
        // Test contra sharedSpheres[i]
    }
}
```

**Beneficio esperado:** Reducción de 30-40% en accesos a memoria global

---

## 3. OPTIMIZACIONES PARA SND_PARALLEL (Prioridad Media-Alta)

### 3.1 Implementar Spatial Partitioning

**Problema Crítico:** Cada píxel testea TODAS las entidades (O(N×M))

**Solución: Grid Spatial Hash**

```glsl
// Pass 1: Culling y extracción (igual que ahora)
// Pass 2: Spatial hash de billboards

layout(local_size_x = 256) in;

struct SpatialGrid {
    uint cellEntityCounts[GRID_X * GRID_Y];
    uint cellEntityIndices[]; // Compact
};

const uint GRID_CELL_SIZE = 32; // Píxeles
const uint GRID_X = (screenResolution.x + GRID_CELL_SIZE - 1) / GRID_CELL_SIZE;
const uint GRID_Y = (screenResolution.y + GRID_CELL_SIZE - 1) / GRID_CELL_SIZE;

void main() {
    uint billboardID = gl_GlobalInvocationID.x;
    if (billboardID >= visibleBillboardCount) return;
    
    SphereBillboard bb = sphereBillboardData[billboardID];
    
    // Calcular celdas que intersecta
    ivec2 cellMin = ivec2(bb.minCorner) / GRID_CELL_SIZE;
    ivec2 cellMax = ivec2(bb.maxCorner) / GRID_CELL_SIZE;
    
    // Asignar a celdas
    for (int cy = cellMin.y; cy <= cellMax.y; cy++) {
        for (int cx = cellMin.x; cx <= cellMax.x; cx++) {
            uint cellIdx = cy * GRID_X + cx;
            uint idx = atomicAdd(spatialGrid.cellEntityCounts[cellIdx], 1);
            spatialGrid.cellEntityIndices[cellIdx * MAX_PER_CELL + idx] = billboardID;
        }
    }
}

// Pass 3: Rendering con spatial lookup
layout(local_size_x = 16, local_size_y = 16) in;

void main() {
    ivec2 pixelCoord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 cellCoord = pixelCoord / GRID_CELL_SIZE;
    uint cellIdx = cellCoord.y * GRID_X + cellCoord.x;
    
    // Solo testear entidades de esta celda (y celdas adyacentes si bbox cruza)
    uint entityCount = spatialGrid.cellEntityCounts[cellIdx];
    
    for (uint i = 0; i < entityCount; i++) {
        uint billboardID = spatialGrid.cellEntityIndices[cellIdx * MAX_PER_CELL + i];
        // Test intersection
    }
}
```

**Beneficio esperado:** Reducción de 100-200x en número de tests (de 20B → 100M)

### 3.2 Optimizar Batching en Shared Memory

**Problema:** Batching actual procesa todas las entidades secuencialmente

**Solución: Early Exit y Depth Sorting**

```glsl
shared SphereBillboard sharedBillboards[256];
shared float sharedDepths[256]; // Pre-computar depths aproximados

void main() {
    // Cargar billboards en shared memory
    loadBillboardsToShared();
    barrier();
    
    // Pre-computar depths y ordenar (bubble sort simple en shared)
    sortBillboardsByDepth();
    barrier();
    
    float minDepth = 1.0;
    vec4 finalColor = vec4(0.0);
    
    // Early exit: si encontramos algo muy cerca, podemos parar
    for (uint i = 0; i < sharedCount; i++) {
        if (sharedDepths[i] > minDepth + EPSILON) {
            break; // Early exit - todo lo demás está más lejos
        }
        
        SphereBillboard bb = sharedBillboards[i];
        if (pixelInBbox(pixelCoord, bb)) {
            float depth = intersectSphere(pixelCoord, bb);
            if (depth < minDepth) {
                minDepth = depth;
                finalColor = shadeSphere(bb, depth);
            }
        }
    }
}
```

**Beneficio esperado:** Reducción de 30-50% en tests por píxel

### 3.3 Reducir Overhead de Dos Passes

**Problema:** Memory barriers entre passes cuestan ~20ms

**Solución: Pipeline Overlap**

```cpp
// Usar múltiples buffers para pipeline
void renderFrame() {
    // Pass 1: Culling frame N (buffer A)
    dispatchCulling(bufferA);
    
    // Pass 2: Spatial hash frame N-1 (buffer B) - mientras Pass 1 corre
    if (frame > 0) {
        dispatchSpatialHash(bufferB);
    }
    
    // Pass 3: Rendering frame N-1 (buffer B)
    if (frame > 0) {
        dispatchRendering(bufferB);
    }
    
    // Swap buffers
    swapBuffers();
}
```

**Beneficio esperado:** Reducción de 15-20ms en overhead

---

## 4. OPTIMIZACIONES PARA CUDA (Prioridad Media)

### 4.1 Optimizar Memory Access Patterns

**Problema:** Accesos no coalesced a memoria global

**Solución: Coalesced Reads**

```cuda
__global__ void sphereRasterKernel(
    const Sphere* __restrict__ spheres,
    uint* __restrict__ depthBuffer,
    vec4* __restrict__ colorBuffer,
    int sphereCount)
{
    // Usar shared memory para coalesced reads
    __shared__ Sphere sharedSpheres[256];
    
    uint tid = threadIdx.x;
    uint bid = blockIdx.x;
    uint entityID = bid * blockDim.x + tid;
    
    // Coalesced load
    if (entityID < sphereCount) {
        sharedSpheres[tid] = spheres[entityID];
    }
    __syncthreads();
    
    // Procesar desde shared memory
    if (entityID < sphereCount) {
        processSphere(sharedSpheres[tid]);
    }
}
```

**Beneficio esperado:** 20-30% mejora en throughput

### 4.2 Usar Texture Memory para Depth Buffer

**Problema:** Accesos aleatorios a depth buffer

**Solución: Texture Memory con Caching**

```cuda
// Bind depth buffer como texture
cudaTextureObject_t depthTex;
cudaResourceDesc resDesc;
cudaTextureDesc texDesc;

// Configurar como texture
cudaCreateTextureObject(&depthTex, &resDesc, &texDesc, NULL);

__device__ float readDepth(cudaTextureObject_t tex, int x, int y) {
    // Texture memory tiene caché automático
    return tex2D<float>(tex, x + 0.5f, y + 0.5f);
}
```

**Beneficio esperado:** 15-25% mejora en hit rate

### 4.3 Optimizar Warp Divergence

**Problema:** Threads en mismo warp procesan entidades con diferentes tamaños de bbox

**Solución: Pre-sort y Group Similar Sizes**

```cuda
// CPU: Ordenar esferas por tamaño de bbox estimado
void sortSpheresByBboxSize(std::vector<Sphere>& spheres) {
    std::sort(spheres.begin(), spheres.end(), [](const Sphere& a, const Sphere& b) {
        // Estimar tamaño de bbox basado en radio y distancia
        float aSize = estimateBboxSize(a);
        float bSize = estimateBboxSize(b);
        return aSize < bSize;
    });
}

// GPU: Procesar en grupos de tamaño similar
__global__ void sphereRasterKernelGrouped(...) {
    // Todos los threads en warp procesan esferas de tamaño similar
    // Menos divergencia
}
```

**Beneficio esperado:** 10-15% mejora en eficiencia de warps

### 4.4 Usar CUDA Streams para Overlap

```cpp
// Crear múltiples streams
cudaStream_t cullingStream, renderingStream;

// Overlap culling y rendering
cudaMemcpyAsync(..., cullingStream);
sphereCullingKernel<<<..., cullingStream>>>(...);

// Mientras culling corre, preparar rendering
cudaMemcpyAsync(..., renderingStream);
sphereRenderingKernel<<<..., renderingStream>>>(...);

// Sincronizar
cudaStreamSynchronize(cullingStream);
cudaStreamSynchronize(renderingStream);
```

**Beneficio esperado:** 10-20% mejora en throughput total

---

## 5. OPTIMIZACIONES COMUNES A TODAS LAS VERSIONES

### 5.1 Mejorar Frustum Culling

**Problema:** Test de 6 planos puede ser optimizado

**Solución: Early Exit y SIMD**

```glsl
bool isSphereInsideOptimized(Sphere sph) {
    vec4 pos = vec4(sph.positionr.xyz, 1.0);
    float r = sph.positionr.w;
    
    // Early exit en primer plano que falla
    if (dot(frustumLeftFace.xyz, pos.xyz) - frustumLeftFace.w < -r) return false;
    if (dot(frustumRightFace.xyz, pos.xyz) - frustumRightFace.w < -r) return false;
    if (dot(frustumNearFace.xyz, pos.xyz) - frustumNearFace.w < -r) return false;
    if (dot(frustumFarFace.xyz, pos.xyz) - frustumFarFace.w < -r) return false;
    if (dot(frustumTopFace.xyz, pos.xyz) - frustumTopFace.w < -r) return false;
    if (dot(frustumBottomFace.xyz, pos.xyz) - frustumBottomFace.w < -r) return false;
    
    return true;
}
```

**Beneficio esperado:** 5-10% mejora en culling

### 5.2 Optimizar Cálculo de Bounding Box 2D

**Problema:** Cálculo de bbox es costoso y se repite

**Solución: Cache y Aproximación**

```glsl
// Cache bbox en frame anterior, solo recalcular si cámara se movió mucho
struct CachedBbox {
    vec2 minCorner;
    vec2 maxCorner;
    float cacheTime;
};

// Aproximación rápida para esferas pequeñas
bboxCorners getSphereBboxFast(Sphere s) {
    vec3 screenPos = projectToScreen(s.positionr.xyz);
    float screenRadius = projectRadius(s.positionr.w, screenPos.z);
    
    // Aproximación: bbox circular → bbox cuadrado
    return bboxCorners(
        screenPos.xy - vec2(screenRadius),
        screenPos.xy + vec2(screenRadius)
    );
}
```

**Beneficio esperado:** 10-15% reducción en tiempo de bbox calculation

### 5.3 Optimizar Ray-Sphere Intersection

**Problema:** Cálculo de intersección puede ser más eficiente

**Solución: Optimizaciones Matemáticas**

```glsl
// Versión optimizada usando FMA
float iSphereOptimized(vec3 ro, vec3 rd, vec3 sph, float radius) {
    vec3 oc = ro - sph;
    float b = dot(oc, rd);
    float ocLenSq = dot(oc, oc);
    float rSq = radius * radius;
    
    // Usar FMA cuando esté disponible
    float h = fma(b, b, -(ocLenSq - rSq));
    
    // Early exit
    if (h < 0.0) return -1.0;
    
    // Fast inverse sqrt para casos donde no necesitamos precisión exacta
    return -b - sqrt(h);
}
```

**Beneficio esperado:** 5-8% mejora en intersection tests

### 5.4 Mejorar Memory Layout

**Problema:** Estructuras no están optimizadas para cache

**Solución: Structure of Arrays (SoA) vs Array of Structures (AoS)**

```cpp
// En lugar de:
struct Sphere {
    vec4 positionr;
    // ... otros campos
};
std::vector<Sphere> spheres; // AoS

// Usar:
struct SphereData {
    float* positionsX;
    float* positionsY;
    float* positionsZ;
    float* radii;
    // ... SoA
};
```

**Beneficio esperado:** 10-20% mejora en cache hit rate

---

## 6. PLAN DE IMPLEMENTACIÓN RECOMENDADO

### Fase 1: Quick Wins (1-2 semanas)
1. ✅ Implementar tile-based rendering en fst_parallel
2. ✅ Optimizar hierarchical Z-buffer en standard_version
3. ✅ Mejorar frustum culling en todas las versiones
4. ✅ Optimizar ray-sphere intersection

**Beneficio esperado:** 20-30% mejora general

### Fase 2: Optimizaciones Medias (2-4 semanas)
1. ✅ Implementar spatial partitioning en snd_parallel
2. ✅ Async compute en standard_version
3. ✅ Memory access optimizations en CUDA
4. ✅ Shared memory optimizations

**Beneficio esperado:** 30-50% mejora adicional

### Fase 3: Optimizaciones Avanzadas (4-8 semanas)
1. ✅ Mesh shaders (si hardware lo soporta)
2. ✅ Variable Rate Shading
3. ✅ Persistent threads
4. ✅ Pipeline overlap completo

**Beneficio esperado:** 20-40% mejora adicional

---

## 7. MÉTRICAS DE ÉXITO

Para cada optimización, medir:

1. **Tiempo de Frame (ms)**
   - Objetivo: Reducir 20-50% según versión

2. **GPU Utilization (%)**
   - Objetivo: >80% active, <20% idle

3. **Cache Hit Rate (%)**
   - Objetivo: >60% L1TEX hit rate

4. **Memory Throughput (GB/s)**
   - Objetivo: Maximizar sin ser cuello de botella

5. **Escalabilidad**
   - Objetivo: Tiempo O(N) o mejor, no O(N²)

---

## 8. CONCLUSIÓN

Las optimizaciones más impactantes son:

1. **Tile-based rendering** para fst_parallel (elimina atomics)
2. **Spatial partitioning** para snd_parallel (elimina O(N×M))
3. **Async compute** para standard_version (reduce idle)
4. **Memory access optimization** para todas las versiones

**Prioridad de implementación:**
1. Fst_parallel: Tile-based rendering (crítico)
2. Snd_parallel: Spatial partitioning (crítico)
3. Standard_version: Async compute + HZB optimization (alto impacto)
4. CUDA: Memory access patterns (medio impacto)

---

_Última actualización: Basado en análisis del código y benchmarks existentes_
