# ==============================================================================
# executables/snd_parallel/CMakeLists.txt - Segunda versión paralela (OpenGL)
# ==============================================================================
# 
# Ejecutables:
#   - snd_cpu_spheres      : Esferas con frustum culling en CPU
#   - snd_cpu_cylinders    : Esferas + cilindros con frustum culling en CPU
#   - snd_gpu_spheres      : Esferas con frustum culling en GPU (compute shader)
#   - snd_gpu_cylinders    : Esferas + cilindros con frustum culling en GPU
#
# ==============================================================================

message(STATUS "Configurando segunda versión paralela (OpenGL compute shaders)...")

set(SND_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin/snd_parallel")

# ------------------------------------------------------------------------------
# CPU Culling - Solo esferas
# ------------------------------------------------------------------------------
add_executable(snd_cpu_spheres
    ${CMAKE_SOURCE_DIR}/executables/snd_parallel/cpu_cull/sphs.cpp
)

target_link_libraries(snd_cpu_spheres PRIVATE 
    lib_core
    lib_parallel
)

target_include_directories(snd_cpu_spheres PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${EXTERN_INCLUDE_DIRS}
)

set_target_properties(snd_cpu_spheres PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${SND_OUTPUT_DIR}/cpu_cull"
)

# ------------------------------------------------------------------------------
# CPU Culling - Con cilindros
# ------------------------------------------------------------------------------
add_executable(snd_cpu_cylinders
    ${CMAKE_SOURCE_DIR}/executables/snd_parallel/cpu_cull/w_cyls.cpp
)

target_link_libraries(snd_cpu_cylinders PRIVATE 
    lib_core
    lib_parallel
)

target_include_directories(snd_cpu_cylinders PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${EXTERN_INCLUDE_DIRS}
)

set_target_properties(snd_cpu_cylinders PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${SND_OUTPUT_DIR}/cpu_cull"
)

# ------------------------------------------------------------------------------
# GPU Culling - Solo esferas
# ------------------------------------------------------------------------------
add_executable(snd_gpu_spheres
    ${CMAKE_SOURCE_DIR}/executables/snd_parallel/gpu_cull/sphs.cpp
)

target_link_libraries(snd_gpu_spheres PRIVATE 
    lib_core
    lib_parallel
)

target_include_directories(snd_gpu_spheres PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${EXTERN_INCLUDE_DIRS}
)

set_target_properties(snd_gpu_spheres PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${SND_OUTPUT_DIR}/gpu_cull"
)

# ------------------------------------------------------------------------------
# GPU Culling - Con cilindros
# ------------------------------------------------------------------------------
add_executable(snd_gpu_cylinders
    ${CMAKE_SOURCE_DIR}/executables/snd_parallel/gpu_cull/w_cyls.cpp
)

target_link_libraries(snd_gpu_cylinders PRIVATE 
    lib_core
    lib_parallel
)

target_include_directories(snd_gpu_cylinders PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${EXTERN_INCLUDE_DIRS}
)

set_target_properties(snd_gpu_cylinders PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${SND_OUTPUT_DIR}/gpu_cull"
)

message(STATUS "  - snd_cpu_spheres")
message(STATUS "  - snd_cpu_cylinders")
message(STATUS "  - snd_gpu_spheres")
message(STATUS "  - snd_gpu_cylinders")
