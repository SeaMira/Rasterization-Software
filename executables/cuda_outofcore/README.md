# Out-of-Core Octree Streaming Pipeline

Pipeline CUDA para renderizado interactivo de datasets atomísticos masivos
que exceden la capacidad de la VRAM, basado en streaming bajo demanda con
coordinación CPU–GPU.

## Fundamentos teóricos

La implementación combina ideas de dos papers:

1. **Sharma et al., "Scalable and portable visualization of large atomistic
   datasets" (CPC 2004)**: Frustum culling jerárquico basado en octree y
   occlusion culling probabilístico basado en densidad y profundidad.

2. **Crassin et al., "GigaVoxels: Ray-Guided Streaming for Efficient and
   Detailed Voxel Rendering" (I3D 2009)**: Pool de bloques con LRU,
   doble buffering y coordinación CPU–GPU para streaming.

## Arquitectura

```
┌──────────────────────────────────────────────────────────┐
│  PREPROCESAMIENTO (una vez, escena estática)             │
│                                                          │
│  Átomos ──→ Morton codes ──→ Sort ──→ Bloques (~512)    │
│         ──→ Block file (.bin) ──→ Octree reducido       │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  POR FRAME                                               │
│                                                          │
│  1. GPU: Frustum culling BFS sobre octree               │
│     → Lista de block IDs visibles                        │
│                                                          │
│  2. GPU: Calcular profundidad + área proyectada          │
│     → thrust::sort por profundidad (front-to-back)       │
│                                                          │
│  3. GPU: Occlusion probabilístico                        │
│     v_c = (1 - D_c) · v_{c-1}, filtrar v_c < umbral     │
│     → Lista filtrada de block IDs                        │
│                                                          │
│  4. GPU: Generar requests (bloques no en pool)           │
│     → thrust::sort + thrust::unique                      │
│     → Readback a CPU                                     │
│                                                          │
│  5. CPU: LRU eviction + carga de disco                   │
│     → cudaMemcpyAsync al write buffer                    │
│                                                          │
│  6. GPU: Construir lista activa de átomos                │
│     → Rasterizar esferas con ray-sphere intersection     │
│                                                          │
│  7. Swap double buffer                                   │
└──────────────────────────────────────────────────────────┘
```

## Estructura de archivos

```
include/outofcore/
  outofcore_types.h      – Tipos compartidos (nodos, bloques, constantes)
  morton.h               – Codificación Morton 3D (CPU + device)
  block_file_io.h        – Lectura/escritura del archivo de bloques
  octree_builder.h       – Construcción del octree reducido
  preprocessor.h         – Pipeline de preprocesamiento completo
  streaming_manager.h    – LRU + doble buffering CPU-side

src/outofcore/
  block_file_io.cpp      – Implementación de I/O binario
  octree_builder.cpp     – Construcción recursiva del octree
  preprocessor.cpp       – Morton sort → bloques → octree
  streaming_manager.cpp  – Gestión del pool con LRU

kernels/outofcore/
  outofcore_kernels.cu   – Kernels: frustum cull, depth/area, occlusion,
                           requests, lista activa
  screen_clear_ooc.cu    – Limpieza de pantalla
  sphere_raster_ooc.cu   – Rasterización de esferas (ray-sphere)

executables/cuda_outofcore/
  outofcore_pipeline.cpp – Ejecutable principal (render loop)
  CMakeLists.txt         – Target de CMake
```

## Formato del archivo de bloques

```
[OocBlockFileHeader]
  magic        (uint64)  – 0x4F4F43424C4B4454 ("OOCBLKDT")
  numBlocks    (uint32)
  atomsPerBlock(uint32)
  sceneMin     (vec3 + pad)
  sceneMax     (vec3 + pad)

[Block 0]
  aabbMin    (vec3)
  aabbMax    (vec3)
  atomCount  (uint32)
  atoms[atomCount] (vec4 cada uno: x, y, z, radius)

[Block 1]
  ...
```

## Compilación

```bash
cmake -B build -DBUILD_CUDA_VERSION=ON
cmake --build build --target cuda_outofcore_pipeline
```

## Modo verbose (construcción del octree paso a paso)

Para observar la construcción del octree durante el preprocesamiento, ejecuta con `-v` o `--verbose`:

```bash
./cuda_outofcore_pipeline -v
# En Windows: cuda_outofcore_pipeline.exe -v
```

### Visual Studio

1. Clic derecho en el proyecto **cuda_outofcore_pipeline** → **Propiedades**
2. **Configuración** → **Depuración** → **Argumentos de comando**
3. Escribe `-v` en el campo
4. Ejecuta con F5 (depuración) o Ctrl+F5 (sin depurador)

La salida se verá en la ventana de consola al iniciar la aplicación.

### Contenido del modo verbose

- Cada nodo: índice, profundidad, número de bloques
- **Región espacial**: AABB de la celda del octree (`regionMin`, `regionMax`)
- **AABB bloques**: bounding box real de los bloques en ese nodo
- Partición por octante (para nodos interiores)

## Dependencias CUDA utilizadas

| Función                  | Header               | Uso                                    |
|--------------------------|----------------------|----------------------------------------|
| `cudaMemcpyToSymbolAsync`| `cuda_runtime.h`     | Subir constantes a `__constant__`      |
| `cudaMemcpyAsync`        | `cuda_runtime.h`     | Transferencias asíncronas              |
| `cudaHostAlloc`          | `cuda_runtime.h`     | Memoria pinned para readback           |
| `cudaStreamCreate`       | `cuda_runtime.h`     | Streams para paralelismo               |
| `thrust::sort`           | `thrust/sort.h`      | Ordenar bloques por profundidad        |
| `thrust::unique`         | `thrust/unique.h`    | Deduplicar request buffer              |
| `thrust::device_ptr`     | `thrust/device_ptr.h`| Wrapper para punteros device en thrust |
| `atomicAdd`              | device built-in      | Contadores atómicos en kernels         |
| `atomicMin`              | device built-in      | Depth buffer atómico                   |
| `surf2Dwrite`            | `surface_functions.h`| Escritura a surface object (textura)   |

## Streaming con transferencias batched

En `processRequests`, en lugar de un `cudaMemcpyAsync` por bloque, se usa:

1. **Staging buffer**: Todos los bloques se cargan desde disco a un buffer host pinned contiguo.
2. **Una sola transferencia**: `cudaMemcpyAsync` del buffer completo al staging en GPU.
3. **Kernel de scatter**: `launchScatterStagingToPool` dispersa cada bloque al slot correspondiente.
4. **Slot map**: `launchApplySlotMapUpdates` aplica todas las actualizaciones (invalidaciones + nuevas asignaciones) en un solo kernel.

## Occlusion probabilístico

Hay dos implementaciones:

- **`launchProbabilisticOcclusion`**: Un solo thread con la recurrencia secuencial v_c = (1-D_c)*v_{c-1}. Simple y eficiente para listas moderadas.
- **`launchProbabilisticOcclusionDynamicParallelism`**: Usa *dynamic parallelism*: un kernel padre lanza kernels hijos por chunk. Cada hijo procesa un chunk de ~64 bloques. Los hijos se ejecutan secuencialmente (mismo stream). Sirve como ejemplo de dynamic parallelism; para listas grandes, un enfoque con scan paralelo puede ser más rápido.

## Parámetros ajustables

Los siguientes parámetros están en `outofcore_types.h`:

| Parámetro                    | Valor por defecto | Descripción                           |
|------------------------------|-------------------|---------------------------------------|
| `OOC_ATOMS_PER_BLOCK`       | 512               | Átomos por bloque (granularidad)      |
| `OOC_MAX_OCTREE_DEPTH`      | 10                | Profundidad máxima del octree         |
| `OOC_BLOCKS_PER_LEAF`       | 4                 | Bloques máx. por hoja antes de dividir|
| `OOC_MAX_BLOCK_POOL_SLOTS`  | 2048              | Slots en el pool de GPU               |
| `OOC_MAX_REQUESTS_PER_FRAME`| 256               | Bloques a cargar por frame máximo     |
| `OOC_VISIBILITY_THRESHOLD`  | 0.01              | Umbral de visibilidad probabilística  |
