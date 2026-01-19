# Core module source files
# This file lists all source files for the core module

set(CORE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/scene.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/scene_builder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rendering/base_compute_renderer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rendering/render_pipeline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rendering/gpu_cull_renderer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shader/shader_base.cpp
)

# Core module header files (for IDE support)
set(CORE_HEADERS
    ${CMAKE_SOURCE_DIR}/include/core/core.h
    
    # Interfaces
    ${CMAKE_SOURCE_DIR}/include/core/interfaces/i_renderable.h
    ${CMAKE_SOURCE_DIR}/include/core/interfaces/i_cullable.h
    ${CMAKE_SOURCE_DIR}/include/core/interfaces/i_shader.h
    ${CMAKE_SOURCE_DIR}/include/core/interfaces/i_scene_loader.h
    ${CMAKE_SOURCE_DIR}/include/core/interfaces/i_culling_strategy.h
    ${CMAKE_SOURCE_DIR}/include/core/interfaces/i_renderer.h
    
    # Configuration
    ${CMAKE_SOURCE_DIR}/include/core/config/render_config.h
    ${CMAKE_SOURCE_DIR}/include/core/config/scene_config.h
    
    # Scene management
    ${CMAKE_SOURCE_DIR}/include/core/scene/scene.h
    ${CMAKE_SOURCE_DIR}/include/core/scene/scene_builder.h
    
    # Rendering
    ${CMAKE_SOURCE_DIR}/include/core/rendering/base_compute_renderer.h
    ${CMAKE_SOURCE_DIR}/include/core/rendering/render_pipeline.h
    ${CMAKE_SOURCE_DIR}/include/core/rendering/gpu_cull_renderer.h
    
    # Shaders
    ${CMAKE_SOURCE_DIR}/include/core/shader/shader_base.h
)
