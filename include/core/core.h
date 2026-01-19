#ifndef _CORE_H_
#define _CORE_H_

/**
 * @file core.h
 * 
 * @brief Central include file for the core module.
 * 
 * This file provides a convenient single include for all core
 * functionality. Include this file to access:
 * - Configuration classes (RenderConfig, SceneConfig)
 * - Interface definitions (IRenderer, ICullable, etc.)
 * - Scene management (Scene, SceneBuilder)
 * - Rendering components (RenderPipeline, BaseComputeRenderer)
 * - Shader utilities (ShaderBase)
 */

// Interfaces
#include "core/interfaces/i_renderable.h"
#include "core/interfaces/i_cullable.h"
#include "core/interfaces/i_shader.h"
#include "core/interfaces/i_scene_loader.h"
#include "core/interfaces/i_culling_strategy.h"
#include "core/interfaces/i_renderer.h"

// Configuration
#include "core/config/render_config.h"
#include "core/config/scene_config.h"

// Scene management
#include "core/scene/scene.h"
#include "core/scene/scene_builder.h"

// Rendering
#include "core/rendering/base_compute_renderer.h"
#include "core/rendering/render_pipeline.h"

// Shaders
#include "core/shader/shader_base.h"

#endif // _CORE_H_
