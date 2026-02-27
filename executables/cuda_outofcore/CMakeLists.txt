# ==============================================================================
# cuda_outofcore - Out-of-core octree streaming pipeline
# ==============================================================================
# Hierarchical view frustum culling + probabilistic occlusion culling
# with LRU block pool and double-buffered GPU streaming.
#
# Based on:
#   - Sharma et al., CPC 2004 (Atomsviewer)
#   - Crassin et al., I3D 2009 (GigaVoxels)
# ==============================================================================

message(STATUS "Configurando out-of-core CUDA pipeline...")

set(CUDA_OOC_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin/cuda_outofcore")
set(CUDA_OOC_KERNELS_DIR "${CMAKE_SOURCE_DIR}/kernels/outofcore")

set(CUDA_OOC_KERNELS
    "${CUDA_OOC_KERNELS_DIR}/outofcore_kernels.cu"
    "${CUDA_OOC_KERNELS_DIR}/screen_clear_ooc.cu"
    "${CUDA_OOC_KERNELS_DIR}/sphere_raster_ooc.cu"
    "${CUDA_OOC_KERNELS_DIR}/streaming_scatter.cu"
)

set(CUDA_OOC_SOURCES
    "${CMAKE_SOURCE_DIR}/src/outofcore/block_file_io.cpp"
    "${CMAKE_SOURCE_DIR}/src/outofcore/octree_builder.cpp"
    "${CMAKE_SOURCE_DIR}/src/outofcore/preprocessor.cpp"
    "${CMAKE_SOURCE_DIR}/src/outofcore/streaming_manager.cpp"
)

add_executable(cuda_outofcore_pipeline
    ${CMAKE_SOURCE_DIR}/executables/cuda_outofcore/outofcore_pipeline.cpp
    ${CUDA_OOC_KERNELS}
    ${CUDA_OOC_SOURCES}
)

set_source_files_properties(${CUDA_OOC_KERNELS} PROPERTIES LANGUAGE CUDA)

# The main executable uses thrust headers, so it must be compiled as CUDA
set_source_files_properties(
    ${CMAKE_SOURCE_DIR}/executables/cuda_outofcore/outofcore_pipeline.cpp
    PROPERTIES LANGUAGE CUDA
)
# streaming_manager.cpp uses cuda_runtime.h
set_source_files_properties(
    ${CMAKE_SOURCE_DIR}/src/outofcore/streaming_manager.cpp
    PROPERTIES LANGUAGE CUDA
)

target_link_libraries(cuda_outofcore_pipeline PRIVATE
    lib_core
    lib_parallel
    lib_cuda_interop
)

target_include_directories(cuda_outofcore_pipeline PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/kernels
    ${EXTERN_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

configure_cuda_target(cuda_outofcore_pipeline)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(cuda_outofcore_pipeline PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xptxas=-v --maxrregcount=64>
    )
else()
    target_compile_options(cuda_outofcore_pipeline PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-G -lineinfo>
    )
endif()

set_target_properties(cuda_outofcore_pipeline PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CUDA_OOC_OUTPUT_DIR}"
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    CUDA_SEPARABLE_COMPILATION ON
)

# Ensure output directories exist for runtime data
add_custom_command(TARGET cuda_outofcore_pipeline POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/ooc_data"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/media/csv/cuda_outofcore"
)

message(STATUS "  - cuda_outofcore_pipeline")
