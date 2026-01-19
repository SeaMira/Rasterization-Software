/**
 * @file main_refactored.cpp
 * 
 * @brief Example of the refactored application using SOLID principles.
 * 
 * This file demonstrates how to use the new architecture with proper
 * separation of concerns, dependency injection, and configuration
 * management.
 * 
 * Key improvements over the original:
 * - No global variables - all configuration through RenderConfig/SceneConfig
 * - Dependency Injection - Renderer and Scene are injected into pipeline
 * - Single Responsibility - Each class has one clear purpose
 * - Open/Closed - New renderers can be added without modifying existing code
 * - Interface Segregation - Small, focused interfaces
 * - Liskov Substitution - Renderers are interchangeable
 */

#include <iostream>
#include <memory>

// Core includes
#include "core/config/render_config.h"
#include "core/config/scene_config.h"
#include "core/scene/scene.h"
#include "core/scene/scene_builder.h"
#include "core/rendering/render_pipeline.h"
#include "core/rendering/gpu_cull_renderer.h"

// Utility includes
#include "utils/benchmark_resources.h"
#include "utils/scene_config_loader.h"

// Global settings loaded from config file
static SceneSettings g_settings;

/**
 * @brief Creates the render configuration from loaded settings.
 * 
 * Uses the Builder pattern for clean, readable configuration.
 * 
 * @return Configured RenderConfig object.
 */
RenderConfig createRenderConfig()
{
    return RenderConfig::Builder()
        .screenWidth(g_settings.screenWidth)
        .screenHeight(g_settings.screenHeight)
        .title("First Parallel Version: Spheres and Cylinders - GPU Frustum Culling")
        .shown(true)
        .occlusionCulling(g_settings.withOcclusionCulling)
        .workGroupSizePerPixel(g_settings.workGroupSizePerPixelX, g_settings.workGroupSizePerPixelY)
        .workGroupSizePerSphere(g_settings.workGroupSizePerSphere)
        .workGroupSizePerCylinder(g_settings.workGroupSizePerCylinder)
        .downsampleLevel(g_settings.downsampleLevel)
        .build();
}

/**
 * @brief Creates the scene configuration from loaded settings.
 * 
 * @return Configured SceneConfig object.
 */
SceneConfig createSceneConfig()
{
    SceneSourceType sourceType = SceneSourceType::FILE_LOADED;
    if (g_settings.sceneType == "GRID_SCENE") {
        sourceType = SceneSourceType::GRID_GENERATED;
    } else if (g_settings.sceneType == "PACKAGE_SCENE") {
        sourceType = SceneSourceType::PACKED_SCENE;
    }
    
    return SceneConfig::Builder()
        .sourceType(sourceType)
        .filePath(g_settings.getScenePath())
        .sphereCount(g_settings.sphereCount)
        .cylinderCount(g_settings.cylinderCount)
        .buildConfig();
}

/**
 * @brief Loads the scene using the existing utility functions.
 * 
 * This bridges the old loading code with the new Scene class.
 * 
 * @param config Scene configuration.
 * @return Unique pointer to the loaded scene.
 */
std::unique_ptr<Scene> loadScene(const SceneConfig& config)
{
    std::vector<glm::vec4> spheres;
    std::vector<Cylinder> cylinders;
    
    // Use existing function to load data
    getCompleteScene(spheres, config.getSphereCount(), 
                     cylinders, config.getCylinderCount());
    
    // Build scene using the new architecture
    return SceneBuilder()
        .withConfig(config)
        .withSpheresFromVec4(spheres)
        .withCylinders(cylinders)
        .build();
}

/**
 * @brief Gets camera checkpoints from the scene.
 * 
 * @param scene The loaded scene.
 * @return Vector of position/target pairs for benchmarking.
 */
std::vector<std::pair<glm::vec3, glm::vec3>> getSceneCheckpoints(Scene& scene)
{
    // Convert scene data for the existing checkpoint function
    std::vector<glm::vec4> spheresCopy = scene.getSpheres();
    std::vector<Cylinder> cylindersCopy = scene.getCylinders();
    
    return getCheckpoints(
        scene.getSphereCountData(),
        spheresCopy,
        cylindersCopy
    );
}

/**
 * @brief Generates output file paths for profiling.
 * 
 * @param config Scene configuration.
 * @param withOcclusion Whether occlusion culling is enabled.
 * @return Pair of (frame_times_path, process_times_path).
 */
std::pair<std::string, std::string> getProfilerPaths(const SceneConfig& config, bool withOcclusion)
{
    std::string basePath = "media/csv/fst_parallel_w_cyl/gpu_cull/";
    
    if (withOcclusion)
    {
        basePath += "occ_";
    }
    
    if (config.getSourceType() == SceneSourceType::FILE_LOADED)
    {
        basePath += "loaded_scene_" + config.getFileName();
    }
    else
    {
        basePath += "packed_scene";
    }
    
    return {
        basePath + "_frame_times.csv",
        basePath + "_process_times.csv"
    };
}

/**
 * @brief Main application entry point.
 * 
 * Demonstrates the refactored architecture with clean separation of:
 * - Configuration (RenderConfig, SceneConfig)
 * - Data (Scene)
 * - Rendering (GPUCullRenderer)
 * - Orchestration (RenderPipeline)
 */
int main(int argc, char* argv[])
{
    std::cout << "Starting refactored application..." << std::endl;
    
    try
    {
        // Load configuration from file (no recompilation needed!)
        g_settings = SceneConfigLoader::loadDefault();
        
        // Apply settings to global variables for backwards compatibility
        SceneConfigLoader::applyToGlobals(g_settings);
        
        // Create configurations using Builder pattern
        RenderConfig renderConfig = createRenderConfig();
        SceneConfig sceneConfig = createSceneConfig();
        
        // Create render pipeline
        RenderPipeline pipeline(renderConfig);
        
        // Load scene using the new architecture
        auto scene = loadScene(sceneConfig);
        std::cout << "Loaded scene with " 
                  << scene->getSphereCount() << " spheres and "
                  << scene->getCylinderCount() << " cylinders" << std::endl;
        
        // Get checkpoints for benchmarking
        auto checkpoints = getSceneCheckpoints(*scene);
        
        // Enable benchmarking
        pipeline.enableBenchmark(checkpoints);

        // Create and inject the renderer (Dependency Injection)
        auto renderer = std::make_unique<GPUCullRenderer>(g_settings.withOcclusionCulling);
        pipeline.setRenderer(std::move(renderer));

        // Initialize pipeline with scene
        pipeline.initialize(std::move(scene));
        
        // Enable profiling
        auto [frameTimesPath, processTimesPath] = getProfilerPaths(sceneConfig, g_settings.withOcclusionCulling);
        pipeline.enableProfiling(frameTimesPath, processTimesPath, g_settings.timerDuration);
        
        // Optional: Set frame callback for custom per-frame logic
        pipeline.setFrameCallback([](const RenderStatistics& stats) {
            // Custom per-frame logic can go here
            // Return false to stop the application
            return true;
        });
        
        // Run the application
        std::cout << "Running render pipeline..." << std::endl;
        pipeline.run();
        
        std::cout << "Application finished successfully." << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
