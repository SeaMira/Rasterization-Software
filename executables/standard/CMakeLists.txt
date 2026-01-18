# ==============================================================================
# executables/standard/CMakeLists.txt - Pipeline estándar de OpenGL
# ==============================================================================
# 
# Ejecutables:
#   - std_cpu_cull    : Pipeline estándar con frustum culling en CPU
#   - std_gpu_cull    : Pipeline estándar con frustum culling en GPU
#
# ==============================================================================

message(STATUS "Configurando versión estándar del pipeline OpenGL...")

set(STD_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin/standard")

# ------------------------------------------------------------------------------
# CPU Culling
# ------------------------------------------------------------------------------
add_executable(std_cpu_cull
    ${CMAKE_SOURCE_DIR}/executables/standard/cpu_cull/standard_version.cpp
)

target_link_libraries(std_cpu_cull PRIVATE 
    lib_core
    lib_parallel
)

target_include_directories(std_cpu_cull PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${EXTERN_INCLUDE_DIRS}
)

set_target_properties(std_cpu_cull PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${STD_OUTPUT_DIR}/cpu_cull"
)

# ------------------------------------------------------------------------------
# GPU Culling
# ------------------------------------------------------------------------------
add_executable(std_gpu_cull
    ${CMAKE_SOURCE_DIR}/executables/standard/gpu_cull/standard_version.cpp
)

target_link_libraries(std_gpu_cull PRIVATE 
    lib_core
    lib_parallel
)

target_include_directories(std_gpu_cull PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${EXTERN_INCLUDE_DIRS}
)

set_target_properties(std_gpu_cull PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${STD_OUTPUT_DIR}/gpu_cull"
)

message(STATUS "  - std_cpu_cull")
message(STATUS "  - std_gpu_cull")
