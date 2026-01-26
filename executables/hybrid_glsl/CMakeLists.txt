# ==============================================================================
# executables/hybrid_glsl/CMakeLists.txt - Hybrid GLSL Pipeline
# ==============================================================================
# 
# Ejecutables:
#   - hybrid_glsl_pipeline : Hybrid rasterization using OpenGL compute shaders
#
# Strategy:
#   - Small entities: Direct rasterization (1 thread per entity)
#   - Large entities: Tiled rasterization (1 thread per pixel per tile)
#   - Fixed tile size: 256 billboards max per tile
#   - No sorting required (simpler than CUDA version)
#
# ==============================================================================

message(STATUS "Configurando versión híbrida GLSL (OpenGL compute shaders)...")

set(HYBRID_GLSL_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin/hybrid_glsl")

# ------------------------------------------------------------------------------
# Create executable
# ------------------------------------------------------------------------------
add_executable(hybrid_glsl_pipeline
    ${CMAKE_SOURCE_DIR}/executables/hybrid_glsl/hybrid_glsl_pipeline.cpp
)

# ------------------------------------------------------------------------------
# Link libraries
# ------------------------------------------------------------------------------
target_link_libraries(hybrid_glsl_pipeline PRIVATE 
    lib_core
    lib_parallel
)

# ------------------------------------------------------------------------------
# Include directories
# ------------------------------------------------------------------------------
target_include_directories(hybrid_glsl_pipeline PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${EXTERN_INCLUDE_DIRS}
)

# ------------------------------------------------------------------------------
# Output properties
# ------------------------------------------------------------------------------
set_target_properties(hybrid_glsl_pipeline PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${HYBRID_GLSL_OUTPUT_DIR}"
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

message(STATUS "  - hybrid_glsl_pipeline")
