# ==============================================================================
# executables/cuda/CMakeLists.txt - Versiones con CUDA interop
# ==============================================================================
# 
# Ejecutables:
#   - cuda_fst_cylinders   : Primera versión CUDA con esferas y cilindros
#
# ==============================================================================

message(STATUS "Configurando versiones CUDA (interop OpenGL-CUDA)...")

set(CUDA_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin/cuda")
set(CUDA_KERNELS_DIR "${CMAKE_SOURCE_DIR}/kernels")

# ------------------------------------------------------------------------------
# Primera versión CUDA - Esferas y cilindros con GPU culling
# ------------------------------------------------------------------------------
file(GLOB CUDA_FST_KERNELS CONFIGURE_DEPENDS 
    "${CUDA_KERNELS_DIR}/fst/gpu_cull/*.cu"
)

add_executable(cuda_fst_cylinders
    ${CMAKE_SOURCE_DIR}/executables/cuda/gpu_cull/w_cyls.cpp
    ${CUDA_FST_KERNELS}
)

# Marcar archivos .cu como CUDA
set_source_files_properties(${CUDA_FST_KERNELS} PROPERTIES LANGUAGE CUDA)

target_link_libraries(cuda_fst_cylinders PRIVATE 
    lib_core
    lib_parallel
    lib_cuda_interop
)

target_include_directories(cuda_fst_cylinders PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${EXTERN_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

# Configurar propiedades CUDA
configure_cuda_target(cuda_fst_cylinders)

set_target_properties(cuda_fst_cylinders PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CUDA_OUTPUT_DIR}/gpu_cull"
)

message(STATUS "  - cuda_fst_cylinders")
message(STATUS "    Kernels: ${CUDA_FST_KERNELS}")
