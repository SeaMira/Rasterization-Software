# ==============================================================================
# executables/CMakeLists.txt - Configuración de todos los ejecutables
# ==============================================================================
#
# Este archivo organiza los ejecutables por versión/tipo:
#   - Sequential:       Versiones 100% CPU (SDL Renderer)
#   - First Parallel:   Primera versión con OpenGL compute shaders  
#   - Second Parallel:  Segunda versión con OpenGL compute shaders
#   - Standard:         Pipeline estándar de OpenGL
#   - CUDA:             Versiones con interop CUDA-OpenGL
#   - Hybrid GLSL:      Versión híbrida con OpenGL compute shaders
#
# ==============================================================================

# Agregar subdirectorios según las opciones de build activadas
if(BUILD_SEQUENTIAL_VERSION)
    add_subdirectory(sequential)
endif()

if(BUILD_FST_PARALLEL_VERSION)
    add_subdirectory(fst_parallel)
endif()

if(BUILD_SND_PARALLEL_VERSION)
    add_subdirectory(snd_parallel)
endif()

if(BUILD_STANDARD_VERSION)
    add_subdirectory(standard)
endif()

if(BUILD_CUDA_VERSION)
    add_subdirectory(cuda)
    add_subdirectory(cuda_hybrid)
    add_subdirectory(cuda_hybrid_binning)
endif()

# Hybrid GLSL version - uses same dependencies as FST_PARALLEL
if(BUILD_FST_PARALLEL_VERSION)
    add_subdirectory(hybrid_glsl)
endif()
