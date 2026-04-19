# Plan de pruebas: pipeline out-of-core CUDA con oclusión `hiz_probabilistic`

## Restricción de herramientas

**Único perfilador NVIDIA autorizado para este estudio: Nsight Graphics.**  
No se utilizarán Nsight Systems, Nsight Compute ni otras herramientas de la suite Nsight para captura de rendimiento. Las métricas que esas herramientas cubrirían de forma directa se obtendrán **solo en la medida en que Nsight Graphics las exponga** (p. ej. actividad **GPU Trace**, filas de cómputo, contadores PM, correlación con presentes), o mediante **instrumentación propia** del ejecutable (logs, CSV, UI) donde Nsight Graphics no alcance.

---

## 1. Objetivo y alcance

Definir las **pruebas de carga y evaluación** del ejecutable **out-of-core** (`cuda_outofcore_pipeline`), con:

```json
"outofcore": { "occlusion_method": "hiz_probabilistic", ... }
```

en `scene_config.json`.

- No se comparan otras variantes del repositorio (compute shaders, otras ramas CUDA, pipeline estándar).
- Objetivo: caracterizar rendimiento, uso de GPU, memoria y comportamiento del streaming bajo distintas escalas y tipos de escena, con metodología de cámara alineada al estudio previo (centroide, radio \(r\), cinco distancias, varias orientaciones).

### 1.1 Comportamiento de `hiz_probabilistic` (contexto)

1. **Pase HiZ (paralelo por bloque):** prueba frente al HiZ del frame anterior; descarta bloques claramente ocluidos.
2. **Pase probabilístico en mosaico (16×16 tiles):** sobre supervivientes del HiZ, acumulación tipo Atomsviewer por tile.

Implementación: `kernels/outofcore/outofcore_kernels.cu` (`hizProbabilisticPass1Kernel`, `hizProbabilisticPass2TiledKernel`).

---

## 2. Software y configuración fija

| Parámetro | Valor |
|-----------|--------|
| Ejecutable | `cuda_outofcore_pipeline` (Release) |
| Oclusión | `hiz_probabilistic` |
| VSync | Desactivado |
| Programación GPU acelerada por hardware | Activada (Windows / driver) |
| Resolución | Documentar `render.screen_width` / `screen_height` |

Documentar en el informe: `atoms_per_block`, `max_octree_depth`, `blocks_per_leaf`, `max_block_pool_slots`, `max_requests_per_frame`, `visibility_threshold`, `downsample_level`. Si cambia el preprocesado, regenerar `ooc_data/block_data.bin`.

---

## 3. Escenas de simulación

Las pruebas de carga se centran en **una referencia molecular de escala media** y en **escenas tipo grilla** a muy gran escala (millones de esferas). Las moléculas pequeñas del repositorio no son el eje principal del estudio.

### 3.1 Referencia molecular (única)

| ID | Fuente | Esferas (orden de magnitud) |
|----|--------|-----------------------------|
| **MOL** | **8WQL** (`.cif` / empaquetado según `scene_config`) | **~5,5×10⁵** (~medio millón) |

Sirve de ancla frente a las grillas: geometría real, distribución compacta frente a regularidad sintética.

### 3.2 Escalas principales: grilla sintética

Configurar `scene.type` como escena de grilla y `sphere_count` (y dimensiones de grilla) para alcanzar **al menos** el número objetivo de esferas. Documentar `grid.width` × `grid.height` × `grid.depth`, `geometry.separation` y `geometry.sphere_radius`.

| ID | Objetivo de esferas | Notas |
|----|---------------------|--------|
| **G10M** | **10⁷** (10 millones) | Ej. grilla cúbica ~216³ ≈ 1,01×10⁷ |
| **G50M** | **5×10⁷** (50 millones) | Ej. ~368³ ≈ 4,98×10⁷ o ajuste fino a 5×10⁷ |
| **G100M** | **10⁸** (100 millones) | Ej. ~464³ ≈ 9,99×10⁷ |
| **G500M** | **5×10⁸** (500 millones) | Ej. ~794³ ≈ 5,01×10⁸ |

**Advertencia operativa:** preprocesado, RAM de host, tamaño de `block_data.bin` y VRAM crecen con el número de esferas; las escenas G100M/G500M pueden requerir hardware y tiempo de generación acorde.

### 3.3 (Opcional) Escala espacial

Factor global sobre extensión o separación (p. ej. 0.5×, 1×, 2×) solo si se quiere variar densidad espacial manteniendo el recuento.

---

## 4. Trayectoria de cámara y muestreo temporal

1. Eje **Y** vertical.  
2. **Centroide** `mass_center` de las entidades.  
3. \(r_x\), \(r_z\) máximos respecto al centroide en X y Z; \(r = \sqrt{r_x^2 + r_z^2}\).  
4. **Cinco distancias** al eje vertical por el centroide (plano XZ): \(2r/5,\ 4r/5,\ 6r/5,\ 8r/5,\ 2r\) (D1–D5).  
5. **Tres orientaciones** (azimut) por distancia.

### 4.1 Frames por captura (compatible con Nsight Graphics)

Nsight Graphics delimita actividades por **presentes** (límites de frame) y por **duración máxima** del trace. Para alinearse con el estudio previo (~15 frames por posesión):

- En **GPU Trace**, usar condiciones de inicio tras estabilizar la cámara y un **Max Duration** / **Limited To** que cubra **≥ 15 presentes** en esa posesión (o repetir capturas cortas y concatenar análisis).
- Alternativa: **disparo manual** del trace cuando la cámara esté fija y contar presentes en la línea de tiempo hasta completar 15.

**Out-of-core:** aplicar **calentamiento** (\(N\) frames antes del primer trace) o documentar por separado “pool frío” vs “estacionario”.

---

## 5. Métricas y cómo obtenerlas **solo con Nsight Graphics**

### 5.1 Actividad principal: GPU Trace

En la conexión a la aplicación, seleccionar **GPU Trace** (documentación: *GPU Trace Overview* de Nsight Graphics).

Permite, según GPU y versión de la herramienta:

- **Línea de tiempo por cola** (gráficos / cómputo): eventos con marcas de tiempo y duración (draw, dispatch, sincronización entre colas cuando el driver lo expone).
- **Métricas PM** (Performance Monitor) y **muestras de estado de warps** (PC samples): utilización de unidades del GPU, cuellos de botella de throughput, actividad en SM con muestreo temporal (no idéntico a un informe NCU por kernel aislado, pero útil para ocupación efectiva y patrones de stall a nivel de frame).
- **Gráficos de métricas** en la línea de tiempo (*Timeline: Metrics Graphs*): correlación entre uso de SM, memoria, etc., y el tramo del frame.
- En arquitecturas recientes, **Hardware Events** / fila **Compute** puede ofrecer mayor detalle en trabajo de cómputo (CUDA aparece en flujos de cómputo cuando el trace lo captura).

**Tiempo por frame:** medir el intervalo entre **presentes** consecutivos en la línea de tiempo del trace.

**“Tiempos muertos”:** huecos sin trabajo GPU en la cola relevante, o intervalos largos entre submits/presents; correlacionar con sincronizaciones visibles en el trace (p. ej. barreras implícitas tras operaciones pesadas).

### 5.2 OpenGL + CUDA (interop)

La aplicación usa **OpenGL** (presente / FBO) y **CUDA** (kernels, texturas registradas). En GPU Trace:

- Puede aparecer trabajo en colas de **gráficos** y **cómputo**.
- Si algún subconjunto de kernels CUDA no etiqueta la línea de tiempo con el nombre esperado, usar **rangos NVTX** ya insertados en `outofcore_pipeline.cpp` (p. ej. “Octree BFS…”, “Occlusion Culling”, “Compute Block Depth+Area”) si Nsight Graphics los muestra en el informe del trace; si no, anotar la limitación y apoyarse en duración de la fase de cómputo agregada.

### 5.3 Ocupancia y utilización de SM

| Objetivo | Con Nsight Graphics |
|----------|---------------------|
| Utilización de SM / warps en el tiempo | Gráficos de métricas PM y warp-state sampling en GPU Trace (ajustar *PM Bandwidth Limit* y *Warp State Samples* según la guía rápida del propio Nsight Graphics). |
| Detalle por kernel como en Nsight Compute | **No garantizado** con la sola restricción Graphics; donde el trace no desglose el kernel, usar duración del bloque de cómputo y repetición estadística. |

### 5.4 Throughput de memoria

Inferir desde **contadores PM** y vistas de métricas del GPU Trace (memoria global, L2, etc., según lo disponible en la versión del driver y la GPU). No sustituye un perfil NCU por kernel, pero sí tendencias por frame y por fase.

### 5.5 Draw / dispatch

- Contar eventos de **submit** / **dispatch** / **draw** visibles en la línea de tiempo del trace para el frame capturado.
- El raster de esferas OOC es **CUDA**, no draw clásico: el número de “draw calls” puede ser bajo en OpenGL y el coste concentrarse en la cola de cómputo; documentarlo explícitamente en el informe.

### 5.6 Transferencias CPU ↔ GPU

GPU Trace se centra en la **GPU**; las transferencias pueden aparecer indirectamente (huecos, sincronización). Para **tamaño y frecuencia** de `cudaMemcpy`/`Async`:

- Opción dentro de la restricción: **instrumentación propia** (contadores por frame a CSV) sin usar otro perfilador NVIDIA.
- Opción secundaria: lectura de estadísticas que el propio ejecutable ya registre (p. ej. profiler interno), si existe.

### 5.7 Métricas lógicas out-of-core y oclusión (complemento obligatorio)

Como Nsight Graphics no sustituye un contador de negocio de bloques, registrar **en paralelo** la instrumentación del ejecutable:

| Métrica | Descripción |
|---------|-------------|
| `numVisible` | Bloques tras frustum |
| `numFiltered` | Bloques tras `hiz_probabilistic` |
| `numRequests` | Peticiones de streaming por frame |
| `activeCount` | Átomos activos en raster |

**CSV por lotes (sin un I/O por frame):** en `scene_config.json`, dentro de `outofcore`:

- `stats_accumulate_frames` (entero): si es **> 0**, el ejecutable **acumula** esas métricas y el **tiempo de frame** (ms, desde el inicio del cuerpo del frame hasta después de `glFinish`) durante **N frames** consecutivos.
- Al completar el lote, escribe **una sola fila** en `stats_csv_path` (modo **append**): medias y min/max por métrica, índices de frame del lote, `scene_type`, `sphere_count`, `total_blocks`, `pool_slots`. Luego reinicia el acumulador para el siguiente lote (misma sesión, mismas columnas).
- Con `stats_accumulate_frames` = **0** esta salida está desactivada (comportamiento por defecto).

Al cerrar la aplicación, si queda un lote **incompleto**, también se vuelca **una fila** con los frames acumulados hasta ese momento.

Así se vincula **carga lógica** con **forma del frame** en GPU Trace sin penalizar el rendimiento con escrituras CSV por frame.

### 5.8 Preprocesado y VRAM estimada (una fila al iniciar)

Tras cargar la escena y **antes** del bucle de render, el ejecutable puede volcar **una fila** (append) en `outofcore.preprocess_stats_csv_path` (cadena vacía = no escribe fichero; sigue imprimiendo resumen por consola):

| Columna / dato | Significado |
|----------------|-------------|
| Tiempos (ms) | `ms_morton_pipeline` (AABB + Morton + ordenación + reorden de átomos), `ms_blocks_and_file` (partición en bloques + escritura de `block_data.bin`), `ms_octree_build` (octree reducido + `blockIndexBuffer`) |
| `host_structures_bytes` | RAM de los vectores `blocks`, `octreeNodes`, `blockIndexBuffer` en el resultado del preprocess |
| `block_file_bytes` | Tamaño en disco de `block_data.bin` |
| `vram_*` | **Estimación** sumando buffers CUDA conocidos: pipeline OOC (`allocGpuResources`), streaming (pools dobles + staging en GPU + metadatos de scatter), buffer de profundidad CUDA pantalla, textura HiZ (float) si aplica, y **aproximación** RGBA8 del target de color (misma resolución que el render) |
| `vram_sum_estimated_bytes` | Suma de las estimaciones anteriores |
| `cuda_mem_free_bytes` / `cuda_mem_total_bytes` | `cudaMemGetInfo` justo después de la inicialización (memoria libre/total del dispositivo; incluye todo lo que CUDA ve en ese momento, no solo este proceso) |

La VRAM del **interop GL↔CUDA** del color puede tener alineaciones/padding no reflejadas en la estimación; usar la suma como orden de magnitud y contrastar con `cudaMemGetInfo` y Nsight Graphics.

---

## 6. Protocolo de captura (solo Nsight Graphics)

1. Instalar/driver y permisos de contadores según la documentación NVIDIA (*ERR_NVGPUCTRPERM* si aplica).  
2. Release, JSON con `hiz_probabilistic`, VSync off.  
3. Conectar el ejecutable desde Nsight Graphics → **GPU Trace**.  
4. Configurar **Start After** (p. ej. tras \(N\) frames de calentamiento o disparo manual en posesión estable).  
5. Ajustar **Max Duration**, límite de timestamps y presupuesto PM según complejidad (escenas grandes pueden requerir traces más cortos o menor frecuencia de muestreo).  
6. Repetir para cada celda experimental (**MOL** o **G10M|G50M|G100M|G500M** × D1–D5 × 3 ángulos).  
7. Exportar / archivar el **informe de GPU Trace** con nombre explícito, p. ej.:  
   `OOC_HIZPROB_{MOL|G10M|G50M|G100M|G500M}_D{d}_Ang{a}_run{r}.ngfx-gputrace`  
   (extensión exacta según versión de Nsight Graphics).

**Reproducibilidad:** opcionalmente fijar reloj de GPU (`nvidia-smi --lock-gpu-clocks=...`) solo si no entra en conflicto con las políticas del laboratorio; no es Nsight, pero es configuración del sistema.

---

## 7. Matriz experimental

| Dimensión | Niveles |
|-----------|---------|
| Escena | **MOL** (8WQL, ~5×10⁵ esferas) · **G10M** · **G50M** · **G100M** · **G500M** |
| Distancia | D1 … D5 |
| Orientación | 3 por distancia |
| Oclusión | Fija: `hiz_probabilistic` |

Mínimo de posesiones de cámara: **5 escenas × 5 distancias × 3 ángulos = 75**; × ~15 frames por captura ⇒ planificar tiempo y almacenamiento de traces (las escenas de grilla grandes alargan preprocesado y tamaño de datos).

**Repeticiones:** ≥ 3 corridas en celdas críticas (**G100M**, **G500M**, y opcionalmente **MOL**) para mediana / dispersión del tiempo entre presentes.

---

## 8. Entregables

1. Tabla de **duración de frame** (p50 / p95) derivada de intervalos entre presentes en GPU Trace, por escena y distancia.  
2. **Capturas de pantalla o exportaciones** de las secciones relevantes del trace (línea de tiempo + métricas PM) para 1–2 casos representativos por categoría.  
3. Tablas o series derivadas del CSV por lotes (**numVisible / numFiltered / numRequests / activeCount** y tiempo de frame agregado por lote) cruzadas con D1–D5 y escena (MOL / G10M–G500M).  
4. Sección **Limitaciones del método de medición**: solo Nsight Graphics; ausencia de Nsight Systems para correlación CPU detallada; ausencia de Nsight Compute para informes por kernel al estilo NCU; posibles limitaciones de visibilidad de CUDA+GL en un único trace según driver y versión.

---

## 9. Referencias en el repositorio

- `assets/config/scene_config.json` → `outofcore` (`stats_accumulate_frames`, `stats_csv_path`, `preprocess_stats_csv_path`)  
- `kernels/outofcore/outofcore_kernels.cu`  
- `executables/cuda_outofcore/outofcore_pipeline.cpp` (rangos NVTX)  
- Documentación NVIDIA: *Nsight Graphics User Guide* → **GPU Trace Overview**, **GPU Trace UI** (timeline, métricas).

---

*Plan actualizado para cumplir la restricción: perfilado exclusivamente con **Nsight Graphics**; métricas no cubiertas por esta herramienta se obtienen por instrumentación del ejecutable o se declaran como no medibles bajo esta restricción.*
