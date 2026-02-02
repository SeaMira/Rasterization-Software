# ==============================================================================
# cuda_hybrid_binning - Hybrid CUDA pipeline with binning-by-sort (CUB)
# ==============================================================================
# No linked lists; uses CUB DeviceRadixSort + DeviceRunLengthEncode for tile offsets.
# ==============================================================================

message(STATUS "Configurando hybrid binning CUDA...")

set(CUDA_HYBRID_BINNING_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin/cuda_hybrid_binning")
set(CUDA_HYBRID_BINNING_KERNELS_DIR "${CMAKE_SOURCE_DIR}/kernels/hybrid_binning")

set(CUDA_HYBRID_BINNING_KERNELS
    "${CUDA_HYBRID_BINNING_KERNELS_DIR}/frustum_bbox_classify.cu"
    "${CUDA_HYBRID_BINNING_KERNELS_DIR}/generate_pairs.cu"
    "${CUDA_HYBRID_BINNING_KERNELS_DIR}/sort_and_rle.cu"
    "${CUDA_HYBRID_BINNING_KERNELS_DIR}/screen_clear.cu"
    "${CUDA_HYBRID_BINNING_KERNELS_DIR}/small_entity_raster.cu"
    "${CUDA_HYBRID_BINNING_KERNELS_DIR}/tiled_raster_binning.cu"
    "${CUDA_HYBRID_BINNING_KERNELS_DIR}/sphere_pipeline_binning.cu"
    "${CUDA_HYBRID_BINNING_KERNELS_DIR}/cylinder_pipeline_binning.cu"
)

add_executable(cuda_hybrid_binning_pipeline
    ${CMAKE_SOURCE_DIR}/executables/cuda_hybrid_binning/hybrid_binning_pipeline.cpp
    ${CUDA_HYBRID_BINNING_KERNELS}
)

set_source_files_properties(${CUDA_HYBRID_BINNING_KERNELS} PROPERTIES LANGUAGE CUDA)

target_link_libraries(cuda_hybrid_binning_pipeline PRIVATE
    lib_core
    lib_parallel
    lib_cuda_interop
)

target_include_directories(cuda_hybrid_binning_pipeline PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/kernels
    ${EXTERN_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

configure_cuda_target(cuda_hybrid_binning_pipeline)

# target_compile_definitions(cuda_hybrid_binning_pipeline PRIVATE
#     GLM_FORCE_CUDA
#     GLM_ENABLE_EXPERIMENTAL
#     GLM_FORCE_INLINE
# )

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(cuda_hybrid_binning_pipeline PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xptxas=-v --maxrregcount=64>
    )
else()
    target_compile_options(cuda_hybrid_binning_pipeline PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-G -lineinfo>
    )
endif()

set_target_properties(cuda_hybrid_binning_pipeline PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CUDA_HYBRID_BINNING_OUTPUT_DIR}"
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    CUDA_SEPARABLE_COMPILATION ON
)

message(STATUS "  - cuda_hybrid_binning_pipeline")
