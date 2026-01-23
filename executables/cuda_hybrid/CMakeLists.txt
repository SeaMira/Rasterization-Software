# ==============================================================================
# executables/cuda_hybrid/CMakeLists.txt - Hybrid CUDA Pipeline
# ==============================================================================
# 
# Ejecutables:
#   - cuda_hybrid_pipeline : Hybrid rasterization combining:
#       - Small entities: Direct rasterization (1 thread per entity)
#       - Large entities: Tiled rasterization with sorted front-to-back
#
# ==============================================================================

message(STATUS "Configurando versión híbrida CUDA...")

set(CUDA_HYBRID_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin/cuda_hybrid")
set(CUDA_HYBRID_KERNELS_DIR "${CMAKE_SOURCE_DIR}/kernels/hybrid")

# ------------------------------------------------------------------------------
# Collect CUDA kernel sources
# ------------------------------------------------------------------------------
set(CUDA_HYBRID_KERNELS
    # Core kernels
    "${CUDA_HYBRID_KERNELS_DIR}/frustum_bbox_classify.cu"
    "${CUDA_HYBRID_KERNELS_DIR}/radix_sort.cu"
    "${CUDA_HYBRID_KERNELS_DIR}/tile_binning.cu"
    "${CUDA_HYBRID_KERNELS_DIR}/small_entity_raster.cu"
    "${CUDA_HYBRID_KERNELS_DIR}/tiled_raster.cu"
    "${CUDA_HYBRID_KERNELS_DIR}/screen_clear.cu"
    # Separate pipeline implementations
    "${CUDA_HYBRID_KERNELS_DIR}/sphere_pipeline.cu"
    "${CUDA_HYBRID_KERNELS_DIR}/cylinder_pipeline.cu"
)

# ------------------------------------------------------------------------------
# Create executable
# ------------------------------------------------------------------------------
add_executable(cuda_hybrid_pipeline
    ${CMAKE_SOURCE_DIR}/executables/cuda_hybrid/hybrid_pipeline.cpp
    ${CUDA_HYBRID_KERNELS}
)

# Mark .cu files as CUDA
set_source_files_properties(${CUDA_HYBRID_KERNELS} PROPERTIES LANGUAGE CUDA)

# ------------------------------------------------------------------------------
# Link libraries
# ------------------------------------------------------------------------------
target_link_libraries(cuda_hybrid_pipeline PRIVATE 
    lib_core
    lib_parallel
    lib_cuda_interop
)

# ------------------------------------------------------------------------------
# Include directories
# ------------------------------------------------------------------------------
target_include_directories(cuda_hybrid_pipeline PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/kernels
    ${EXTERN_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

# ------------------------------------------------------------------------------
# CUDA configuration
# ------------------------------------------------------------------------------
configure_cuda_target(cuda_hybrid_pipeline)

# Enable CUB (included with CUDA Toolkit)
target_compile_definitions(cuda_hybrid_pipeline PRIVATE
    GLM_FORCE_CUDA
    GLM_ENABLE_EXPERIMENTAL
    GLM_FORCE_INLINE
)

# Additional CUDA flags for optimization
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(cuda_hybrid_pipeline PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3
            --use_fast_math
            -Xptxas=-v
            --maxrregcount=64
        >
    )
else()
    target_compile_options(cuda_hybrid_pipeline PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -G
            -lineinfo
        >
    )
endif()

# ------------------------------------------------------------------------------
# Output properties
# ------------------------------------------------------------------------------
set_target_properties(cuda_hybrid_pipeline PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CUDA_HYBRID_OUTPUT_DIR}"
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    CUDA_SEPARABLE_COMPILATION ON
)

message(STATUS "  - cuda_hybrid_pipeline")
message(STATUS "    Kernels: ${CUDA_HYBRID_KERNELS}")
