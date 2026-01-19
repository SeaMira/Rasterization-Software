# ==============================================================================
# executables/fst_parallel/CMakeLists.txt - Primera versión paralela (OpenGL)
# ==============================================================================
# 
# Ejecutables:
#   - fst_cpu_spheres      : Esferas con frustum culling en CPU
#   - fst_cpu_cylinders    : Esferas + cilindros con frustum culling en CPU
#   - fst_gpu_spheres      : Esferas con frustum culling en GPU (compute shader)
#   - fst_gpu_cylinders    : Esferas + cilindros con frustum culling en GPU
#
# ==============================================================================

message(STATUS "Configurando primera versión paralela (OpenGL compute shaders)...")

set(FST_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin/fst_parallel")

# ------------------------------------------------------------------------------
# CPU Culling - Solo esferas
# ------------------------------------------------------------------------------
add_executable(fst_cpu_spheres
    ${CMAKE_SOURCE_DIR}/executables/fst_parallel/cpu_cull/sphs.cpp
)

target_link_libraries(fst_cpu_spheres PRIVATE 
    lib_core
    lib_parallel
)

target_include_directories(fst_cpu_spheres PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${EXTERN_INCLUDE_DIRS}
)

set_target_properties(fst_cpu_spheres PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${FST_OUTPUT_DIR}/cpu_cull"
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# ------------------------------------------------------------------------------
# CPU Culling - Con cilindros
# ------------------------------------------------------------------------------
add_executable(fst_cpu_cylinders
    ${CMAKE_SOURCE_DIR}/executables/fst_parallel/cpu_cull/w_cyls.cpp
)

target_link_libraries(fst_cpu_cylinders PRIVATE 
    lib_core
    lib_parallel
)

target_include_directories(fst_cpu_cylinders PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${EXTERN_INCLUDE_DIRS}
)

set_target_properties(fst_cpu_cylinders PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${FST_OUTPUT_DIR}/cpu_cull"
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# ------------------------------------------------------------------------------
# GPU Culling - Solo esferas
# ------------------------------------------------------------------------------
add_executable(fst_gpu_spheres
    ${CMAKE_SOURCE_DIR}/executables/fst_parallel/gpu_cull/sphs.cpp
)

target_link_libraries(fst_gpu_spheres PRIVATE 
    lib_core
    lib_parallel
)

target_include_directories(fst_gpu_spheres PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${EXTERN_INCLUDE_DIRS}
)

set_target_properties(fst_gpu_spheres PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${FST_OUTPUT_DIR}/gpu_cull"
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# ------------------------------------------------------------------------------
# GPU Culling - Con cilindros
# ------------------------------------------------------------------------------
add_executable(fst_gpu_cylinders
    ${CMAKE_SOURCE_DIR}/executables/fst_parallel/gpu_cull/w_cyls.cpp
)

target_link_libraries(fst_gpu_cylinders PRIVATE 
    lib_core
    lib_parallel
)

target_include_directories(fst_gpu_cylinders PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${EXTERN_INCLUDE_DIRS}
)

set_target_properties(fst_gpu_cylinders PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${FST_OUTPUT_DIR}/gpu_cull"
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# ------------------------------------------------------------------------------
# GPU Culling - Con cilindros (Indirect Dispatch)
# ------------------------------------------------------------------------------
# Esta versión separa el frustum culling de la rasterización en shaders
# diferentes, utilizando Indirect Dispatch para ejecutar el shader de
# rasterización solo con el número necesario de work groups.
# ------------------------------------------------------------------------------
add_executable(fst_gpu_cylinders_indirect
    ${CMAKE_SOURCE_DIR}/executables/fst_parallel/gpu_cull_indirect/w_cyls.cpp
)

target_link_libraries(fst_gpu_cylinders_indirect PRIVATE 
    lib_core
    lib_parallel
)

target_include_directories(fst_gpu_cylinders_indirect PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${EXTERN_INCLUDE_DIRS}
)

set_target_properties(fst_gpu_cylinders_indirect PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${FST_OUTPUT_DIR}/gpu_cull_indirect"
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

message(STATUS "  - fst_cpu_spheres")
message(STATUS "  - fst_cpu_cylinders")
message(STATUS "  - fst_gpu_spheres")
message(STATUS "  - fst_gpu_cylinders")
message(STATUS "  - fst_gpu_cylinders_indirect")
