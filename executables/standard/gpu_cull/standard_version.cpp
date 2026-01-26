/**
 * @file standard_version.cpp
 * 
 * @brief Standard OpenGL rendering pipeline with GPU frustum culling.
 * 
 * This file implements a traditional OpenGL rendering pipeline using
 * vertex/fragment/geometry shaders with optional occlusion culling.
 * Refactored following SOLID principles for better maintainability.
 * 
 * Key features:
 * - GPU-based frustum culling via compute shaders
 * - Optional Hi-Z occlusion culling
 * - Traditional rasterization pipeline for final rendering
 * - Configuration-driven design (no hardcoded values)
 * 
 * Architecture:
 * - Configuration loading through SceneConfigLoader
 * - Builder pattern for render/scene configuration
 * - Separation of concerns between rendering modes
 * - Clean resource management with RAII
 */

#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <unordered_map>
#include <chrono>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glad/glad.h>
#include <SDL3/SDL.h>

#include "appSDLGL.h"
#include "utils/parallel/aux_functions.h"
#include "utils/benchmark_resources.h"
#include "utils/scene_config_loader.h"
#include "ux/input.h"
#include "ux/camera_controller.h"
#include "ux/cinematic/benchmark.h"
#include "ux/profiler/profiler.h"
#include "vis/gl/frame_buffer.h"
#include "vis/gl/storage_buffer.h"
#include "vis/gl/texture.h"
#include "vis/canvas.h"
#include "vis/compute_shader_program.h"
#include "vis/shader_program.h"
#include "core/config/render_config.h"
#include "core/config/scene_config.h"
#include "geometry/sphere/sphere.h"
#include "geometry/cylinder/cylinder.h"
#include "algorithms/frustum_cull.h"

// ============================================================================
// Forward Declarations
// ============================================================================

class StandardRenderer;
class StandardRendererWithOcclusion;
class StandardRendererWithoutOcclusion;

// ============================================================================
// Configuration Structures
// ============================================================================

/**
 * @struct RenderState
 * @brief Encapsulates mutable render state for a frame.
 */
struct RenderState
{
    int visibleSpheresCount = 0;
    int notOccludedSpheresCount = 0;
    int visibleCylindersCount = 0;
    int notOccludedCylindersCount = 0;
};

// ============================================================================
// Global Configuration (loaded from file)
// ============================================================================

static SceneSettings g_settings;
static std::chrono::steady_clock::time_point g_startTime;

// ============================================================================
// Configuration Functions
// ============================================================================

/**
 * @brief Creates render configuration from loaded settings.
 * 
 * Uses the Builder pattern for clean, readable configuration.
 * 
 * @return Configured RenderConfig object.
 */
RenderConfig createRenderConfig()
{
    const std::string title = "Standard OpenGL Version: Spheres and Cylinders - GPU Frustum Culling";
    const std::string suffix = g_settings.withOcclusionCulling 
        ? " - With Occlusion Culling" 
        : " - Without Occlusion Culling";
    
    return RenderConfig::Builder()
        .screenWidth(g_settings.screenWidth)
        .screenHeight(g_settings.screenHeight)
        .title(title + suffix)
        .shown(true)
        .occlusionCulling(g_settings.withOcclusionCulling)
        .workGroupSizePerPixel(g_settings.workGroupSizePerPixelX, g_settings.workGroupSizePerPixelY)
        .workGroupSizePerSphere(g_settings.workGroupSizePerSphere)
        .workGroupSizePerCylinder(g_settings.workGroupSizePerCylinder)
        .downsampleLevel(g_settings.downsampleLevel)
        .build();
}

/**
 * @brief Creates scene configuration from loaded settings.
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
 * @brief Generates profiler output paths based on configuration.
 * 
 * @param sceneConfig Scene configuration for path generation.
 * @param withOcclusion Whether occlusion culling is enabled.
 * @return Pair of (frame_times_path, process_times_path).
 */
std::pair<std::string, std::string> getProfilerPaths(
    const SceneConfig& sceneConfig, 
    bool withOcclusion)
{
    std::string basePath = "media/csv/standard_version/gpu_cull/";
    
    if (withOcclusion) {
        basePath += "occ_";
    }
    
    if (sceneConfig.getSourceType() == SceneSourceType::FILE_LOADED) {
        basePath += "loaded_scene_" + sceneConfig.getFileName();
    } else {
        basePath += "packed_scene";
    }
    
    return {
        basePath + "_frame_times.csv",
        basePath + "_process_times.csv"
    };
}

// ============================================================================
// Shader Path Utilities
// ============================================================================

/**
 * @brief Gets the base shader path for standard version shaders.
 * 
 * @return Base path string for shader files.
 */
inline std::string getShaderBasePath()
{
    return "assets/shaders/standard_version/gpu_cull/";
}

/**
 * @brief Creates a full shader path.
 * 
 * @param shaderName Name of the shader file.
 * @return Complete path to the shader.
 */
inline std::string getShaderPath(const std::string& shaderName)
{
    return getShaderBasePath() + shaderName;
}

// ============================================================================
// Resource Creation Functions
// ============================================================================

/**
 * @brief Creates the scene info data map for UI display.
 * 
 * @param config Render configuration.
 * @param state Mutable render state.
 * @param sphereCount Reference to sphere count.
 * @param cylinderCount Reference to cylinder count.
 * @return Map of scene data for UI.
 */
std::unordered_map<std::string, int*> createSceneInfoMap(
    const RenderConfig& config,
    RenderState& state,
    int& sphereCount,
    int& cylinderCount)
{
    // Note: We need mutable references for the UI to update
    static int screenWidth, screenHeight;
    screenWidth = config.getScreenWidth();
    screenHeight = config.getScreenHeight();
    
    return {
        {"Screen width", &screenWidth},
        {"Screen height", &screenHeight},
        {"Sphere count", &sphereCount},
        {"Spheres On Frustum", &state.visibleSpheresCount},
        {"Drawn spheres", &state.notOccludedSpheresCount},
        {"Cylinder count", &cylinderCount},
        {"Cylinders On Frustum", &state.visibleCylindersCount},
        {"Drawn cylinders", &state.notOccludedCylindersCount}
    };
}

/**
 * @brief Creates the framebuffer setup for rendering.
 * 
 * @param config Render configuration.
 * @param withOcclusion Whether to include occlusion-related textures.
 * @return Configured Framebuffer.
 */
struct FramebufferResources
{
    std::unique_ptr<Texture> imageTexture;
    std::unique_ptr<Texture> pixelIdTexture;
    std::unique_ptr<Texture> depthTexture;
    std::unique_ptr<Texture> downsampledDepthTexture;
    std::unique_ptr<Framebuffer> fbo;
    
    // Enable move semantics (unique_ptr is move-only)
    FramebufferResources() = default;
    FramebufferResources(FramebufferResources&&) noexcept = default;
    FramebufferResources& operator=(FramebufferResources&&) noexcept = default;
    
    // Disable copy
    FramebufferResources(const FramebufferResources&) = delete;
    FramebufferResources& operator=(const FramebufferResources&) = delete;
    
    void bind() const { glBindFramebuffer(GL_FRAMEBUFFER, fbo->getId()); }
    void bindAsReadFramebuffer() const { glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo->getId()); }
};

/**
 * @brief Creates framebuffer resources with occlusion support.
 * 
 * @param config Render configuration.
 * @return FramebufferResources struct with all textures and FBO.
 */
FramebufferResources createFramebufferWithOcclusion(const RenderConfig& config)
{
    FramebufferResources resources;
    
    const int width = config.getScreenWidth();
    const int height = config.getScreenHeight();
    const int downsampleLevel = config.getDownsampleLevel();
    
    resources.imageTexture = std::make_unique<Texture>(
        GL_TEXTURE_2D, GL_RGBA8, width, height, 0);
    
    resources.pixelIdTexture = std::make_unique<Texture>(
        GL_TEXTURE_2D, GL_R32UI, width, height, 
        GL_RED_INTEGER, GL_UNSIGNED_INT, 0, nullptr);
    
    resources.depthTexture = std::make_unique<Texture>(
        GL_TEXTURE_2D, GL_DEPTH_COMPONENT32F, width, height, 0);
    
    resources.downsampledDepthTexture = std::make_unique<Texture>(
        GL_TEXTURE_2D, GL_R32F, 
        width / (1 << downsampleLevel), 
        height / (1 << downsampleLevel), 
        1, downsampleLevel);
    
    resources.fbo = std::make_unique<Framebuffer>();
    resources.fbo->attachTexture(GL_COLOR_ATTACHMENT0, *resources.imageTexture);
    resources.fbo->attachTexture(GL_COLOR_ATTACHMENT1, *resources.pixelIdTexture);
    resources.fbo->attachTexture(GL_DEPTH_ATTACHMENT, *resources.depthTexture);
    
    // Setup multiple render targets
    GLenum drawBuffers[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    glBindFramebuffer(GL_FRAMEBUFFER, resources.fbo->getId());
    glDrawBuffers(2, drawBuffers);
    
    return resources;
}

/**
 * @brief Creates framebuffer resources without occlusion support.
 * 
 * @param config Render configuration.
 * @return FramebufferResources struct with basic textures and FBO.
 */
FramebufferResources createFramebufferWithoutOcclusion(const RenderConfig& config)
{
    FramebufferResources resources;
    
    const int width = config.getScreenWidth();
    const int height = config.getScreenHeight();
    const int mipLevels = 1 + static_cast<int>(
        std::floor(std::log2(std::max(width, height))));
    
    resources.imageTexture = std::make_unique<Texture>(
        GL_TEXTURE_2D, GL_RGBA8, width, height, 0);
    
    resources.depthTexture = std::make_unique<Texture>(
        GL_TEXTURE_2D, GL_DEPTH_COMPONENT32F, width, height, 0, mipLevels);
    
    resources.fbo = std::make_unique<Framebuffer>();
    resources.fbo->attachTexture(GL_COLOR_ATTACHMENT0, *resources.imageTexture);
    resources.fbo->attachTexture(GL_DEPTH_ATTACHMENT, *resources.depthTexture);
    
    return resources;
}

// ============================================================================
// Shader Resources
// ============================================================================

/**
 * @struct ShaderResources
 * @brief Encapsulates all shaders needed for rendering.
 */
struct ShaderResources
{
    std::unique_ptr<ComputeShader> spheresCullingShader;
    std::unique_ptr<ShaderProgram> spheresShader;
    std::unique_ptr<ComputeShader> cylindersCullingShader;
    std::unique_ptr<ShaderProgram> cylindersShader;
    
    // Occlusion-specific shaders
    std::unique_ptr<ComputeShader> pixelCountShader;
    std::unique_ptr<ComputeShader> hizPyramidShader;
    
    // Enable move semantics (unique_ptr is move-only)
    ShaderResources() = default;
    ShaderResources(ShaderResources&&) noexcept = default;
    ShaderResources& operator=(ShaderResources&&) noexcept = default;
    
    // Disable copy
    ShaderResources(const ShaderResources&) = delete;
    ShaderResources& operator=(const ShaderResources&) = delete;
};

/**
 * @brief Creates shaders for occlusion culling mode.
 * 
 * @return ShaderResources with occlusion shaders.
 */
ShaderResources createShadersWithOcclusion()
{
    ShaderResources shaders;
    const std::string basePath = getShaderBasePath();
    
    shaders.pixelCountShader = std::make_unique<ComputeShader>(
        (basePath + "pixel_count.compute").c_str(), "Pixel Count Shader");
    
    shaders.hizPyramidShader = std::make_unique<ComputeShader>(
        (basePath + "mipmap_gen.compute").c_str(), "Hiz Pyramid Shader");
    
    shaders.spheresCullingShader = std::make_unique<ComputeShader>(
        (basePath + "spheresFrusOccCulling.compute").c_str(), "Spheres Culling Shader");
    
    shaders.spheresShader = std::make_unique<ShaderProgram>(
        (basePath + "spheres.vert").c_str(),
        (basePath + "spheresOccCulling.frag").c_str(),
        (basePath + "spheresOccCulling.geom").c_str(),
        "Spheres With Occ Culling Program",
        "Spheres Vertex Shader",
        "Spheres Fragment Shader",
        "Spheres Geometry Shader");
    shaders.spheresShader->linkProgram();
    
    shaders.cylindersCullingShader = std::make_unique<ComputeShader>(
        (basePath + "cylindersFrusOccCulling.compute").c_str(), "Cylinders Culling Shader");
    
    shaders.cylindersShader = std::make_unique<ShaderProgram>(
        (basePath + "cylinders.vert").c_str(),
        (basePath + "cylindersOccCulling.frag").c_str(),
        (basePath + "cylindersOccCulling.geom").c_str(),
        "Cylinders Program",
        "Cylinders Vertex Shader",
        "Cylinders Fragment Shader",
        "Cylinders Geometry Shader");
    shaders.cylindersShader->linkProgram();
    
    return shaders;
}

/**
 * @brief Creates shaders for non-occlusion mode.
 * 
 * @return ShaderResources without occlusion shaders.
 */
ShaderResources createShadersWithoutOcclusion()
{
    ShaderResources shaders;
    const std::string basePath = getShaderBasePath();
    
    shaders.spheresCullingShader = std::make_unique<ComputeShader>(
        (basePath + "spheresFrusCulling.compute").c_str(), "Spheres Culling Shader");
    
    shaders.spheresShader = std::make_unique<ShaderProgram>(
        (basePath + "spheres.vert").c_str(),
        (basePath + "spheres.frag").c_str(),
        (basePath + "spheres.geom").c_str(),
        "Spheres Program",
        "Spheres Vertex Shader",
        "Spheres Fragment Shader",
        "Spheres Geometry Shader");
    shaders.spheresShader->linkProgram();
    
    shaders.cylindersCullingShader = std::make_unique<ComputeShader>(
        (basePath + "cylindersFrusCulling.compute").c_str(), "Cylinders Culling Shader");
    
    shaders.cylindersShader = std::make_unique<ShaderProgram>(
        (basePath + "cylinders.vert").c_str(),
        (basePath + "cylinders.frag").c_str(),
        (basePath + "cylinders.geom").c_str(),
        "Cylinders Program",
        "Cylinders Vertex Shader",
        "Cylinders Fragment Shader",
        "Cylinders Geometry Shader");
    shaders.cylindersShader->linkProgram();
    
    return shaders;
}

// ============================================================================
// Buffer Resources
// ============================================================================

/**
 * @struct BufferResources
 * @brief Encapsulates all GPU buffers for rendering.
 */
struct BufferResources
{
    std::unique_ptr<StorageBuffer> sphereBuffer;
    std::unique_ptr<StorageBuffer> visibleSpheresBuffer;
    std::unique_ptr<StorageBuffer> cylinderBuffer;
    std::unique_ptr<StorageBuffer> visibleCylindersBuffer;
    std::unique_ptr<StorageBuffer> visibleEntitiesCounter;
    std::unique_ptr<StorageBuffer> frustumEntitiesCounter;
    std::unique_ptr<StorageBuffer> visibilityFramesBuffer;
    GLuint emptyVAO = 0;
    
    // Default constructor
    BufferResources() = default;
    
    // Move constructor
    BufferResources(BufferResources&& other) noexcept
        : sphereBuffer(std::move(other.sphereBuffer))
        , visibleSpheresBuffer(std::move(other.visibleSpheresBuffer))
        , cylinderBuffer(std::move(other.cylinderBuffer))
        , visibleCylindersBuffer(std::move(other.visibleCylindersBuffer))
        , visibleEntitiesCounter(std::move(other.visibleEntitiesCounter))
        , frustumEntitiesCounter(std::move(other.frustumEntitiesCounter))
        , visibilityFramesBuffer(std::move(other.visibilityFramesBuffer))
        , emptyVAO(other.emptyVAO)
    {
        other.emptyVAO = 0; // Prevent double deletion
    }
    
    // Move assignment operator
    BufferResources& operator=(BufferResources&& other) noexcept
    {
        if (this != &other)
        {
            // Clean up existing resources
            if (emptyVAO != 0) {
                glDeleteVertexArrays(1, &emptyVAO);
            }
            
            // Move resources
            sphereBuffer = std::move(other.sphereBuffer);
            visibleSpheresBuffer = std::move(other.visibleSpheresBuffer);
            cylinderBuffer = std::move(other.cylinderBuffer);
            visibleCylindersBuffer = std::move(other.visibleCylindersBuffer);
            visibleEntitiesCounter = std::move(other.visibleEntitiesCounter);
            frustumEntitiesCounter = std::move(other.frustumEntitiesCounter);
            visibilityFramesBuffer = std::move(other.visibilityFramesBuffer);
            emptyVAO = other.emptyVAO;
            other.emptyVAO = 0;
        }
        return *this;
    }
    
    // Disable copy
    BufferResources(const BufferResources&) = delete;
    BufferResources& operator=(const BufferResources&) = delete;
    
    ~BufferResources()
    {
        if (emptyVAO != 0) {
            glDeleteVertexArrays(1, &emptyVAO);
        }
    }
};

/**
 * @brief Creates buffer resources for occlusion mode.
 * 
 * @param spheres Sphere data.
 * @param cylinders Cylinder data.
 * @return Configured BufferResources.
 */
BufferResources createBuffersWithOcclusion(
    const std::vector<Sphere>& spheres,
    const std::vector<Cylinder>& cylinders)
{
    BufferResources buffers;
    const int sphereCount = static_cast<int>(spheres.size());
    const int cylinderCount = static_cast<int>(cylinders.size());
    
    std::cout << "Creating sphere buffer..." << std::endl;
    buffers.sphereBuffer = std::make_unique<StorageBuffer>(
        GL_SHADER_STORAGE_BUFFER, 
        static_cast<int>(sphereCount * sizeof(Sphere)), 1,
        spheres.data(), GL_STATIC_DRAW);
    
    std::cout << "Creating sphere indices buffer..." << std::endl;
    std::vector<GLuint> visibleSpheres(sphereCount, 0);
    buffers.visibleSpheresBuffer = std::make_unique<StorageBuffer>(
        GL_SHADER_STORAGE_BUFFER,
        static_cast<int>(sphereCount * sizeof(GLuint)), 2,
        visibleSpheres.data(), GL_STATIC_DRAW);
    
    GLuint zero = 0;
    buffers.visibleEntitiesCounter = std::make_unique<StorageBuffer>(
        GL_SHADER_STORAGE_BUFFER, static_cast<int>(sizeof(GLuint)), 3,
        &zero, GL_STATIC_DRAW);
    
    std::cout << "Creating cylinder buffer..." << std::endl;
    buffers.cylinderBuffer = std::make_unique<StorageBuffer>(
        GL_SHADER_STORAGE_BUFFER,
        static_cast<int>(cylinderCount * sizeof(Cylinder)), 4,
        cylinders.data(), GL_STATIC_DRAW);
    
    std::cout << "Creating cylinder indices buffer..." << std::endl;
    std::vector<GLuint> visibleCylinders(cylinderCount, 0);
    buffers.visibleCylindersBuffer = std::make_unique<StorageBuffer>(
        GL_SHADER_STORAGE_BUFFER,
        static_cast<int>(cylinderCount * sizeof(GLuint)), 5,
        visibleCylinders.data(), GL_STATIC_DRAW);
    
    // Visibility frames buffer for occlusion culling
    const int totalEntities = sphereCount + cylinderCount;
    std::vector<GLuint> visibilityFrames(2 * totalEntities, 10);
    buffers.visibilityFramesBuffer = std::make_unique<StorageBuffer>(
        GL_SHADER_STORAGE_BUFFER,
        static_cast<int>(2 * totalEntities * sizeof(GLuint)), 6,
        visibilityFrames.data(), GL_DYNAMIC_COPY);
    
    buffers.frustumEntitiesCounter = std::make_unique<StorageBuffer>(
        GL_SHADER_STORAGE_BUFFER, static_cast<int>(sizeof(GLuint)), 7,
        &zero, GL_STATIC_DRAW);
    
    glCreateVertexArrays(1, &buffers.emptyVAO);
    
    return buffers;
}

/**
 * @brief Creates buffer resources for non-occlusion mode.
 * 
 * @param spheres Sphere data.
 * @param cylinders Cylinder data.
 * @return Configured BufferResources.
 */
BufferResources createBuffersWithoutOcclusion(
    const std::vector<Sphere>& spheres,
    const std::vector<Cylinder>& cylinders)
{
    BufferResources buffers;
    const int sphereCount = static_cast<int>(spheres.size());
    const int cylinderCount = static_cast<int>(cylinders.size());
    
    std::cout << "Creating sphere buffer..." << std::endl;
    buffers.sphereBuffer = std::make_unique<StorageBuffer>(
        GL_SHADER_STORAGE_BUFFER,
        static_cast<int>(sphereCount * sizeof(Sphere)), 1,
        spheres.data(), GL_STATIC_DRAW);
    
    std::cout << "Creating sphere indices buffer..." << std::endl;
    std::vector<GLuint> visibleSpheres(sphereCount, 0);
    buffers.visibleSpheresBuffer = std::make_unique<StorageBuffer>(
        GL_SHADER_STORAGE_BUFFER,
        static_cast<int>(sphereCount * sizeof(GLuint)), 2,
        visibleSpheres.data(), GL_STATIC_DRAW);
    
    GLuint zero = 0;
    buffers.visibleEntitiesCounter = std::make_unique<StorageBuffer>(
        GL_SHADER_STORAGE_BUFFER, static_cast<int>(sizeof(GLuint)), 3,
        &zero, GL_STATIC_DRAW);
    
    std::cout << "Creating cylinder buffer..." << std::endl;
    buffers.cylinderBuffer = std::make_unique<StorageBuffer>(
        GL_SHADER_STORAGE_BUFFER,
        static_cast<int>(cylinderCount * sizeof(Cylinder)), 4,
        cylinders.data(), GL_STATIC_DRAW);
    
    std::cout << "Creating cylinder indices buffer..." << std::endl;
    std::vector<GLuint> visibleCylinders(cylinderCount, 0);
    buffers.visibleCylindersBuffer = std::make_unique<StorageBuffer>(
        GL_SHADER_STORAGE_BUFFER,
        static_cast<int>(cylinderCount * sizeof(GLuint)), 5,
        visibleCylinders.data(), GL_STATIC_DRAW);
    
    glCreateVertexArrays(1, &buffers.emptyVAO);
    
    return buffers;
}

// ============================================================================
// Render Loop Functions
// ============================================================================

/**
 * @brief Prepares the frame for rendering.
 * 
 * @param fbo Framebuffer resources.
 * @param config Render configuration.
 */
void prepareFrame(const FramebufferResources& fbo, const RenderConfig& config)
{
    glBindFramebuffer(GL_FRAMEBUFFER, fbo.fbo->getId());
    glViewport(0, 0, config.getScreenWidth(), config.getScreenHeight());
    glClearColor(1.0f, 1.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
}

/**
 * @brief Performs Hi-Z pyramid generation for occlusion culling.
 * 
 * @param shaders Shader resources.
 * @param fbo Framebuffer resources.
 * @param config Render configuration.
 */
void performHiZPyramidGeneration(
    const ShaderResources& shaders,
    const FramebufferResources& fbo,
    const RenderConfig& config)
{
    if (!shaders.hizPyramidShader) return;
    
    fbo.depthTexture->bind();
    fbo.downsampledDepthTexture->bindImage(GL_READ_WRITE, GL_R32F);
    
    shaders.hizPyramidShader->use();
    
    const glm::vec2 texelDimensions(
        1.0f / static_cast<float>(config.getScreenWidth()),
        1.0f / static_cast<float>(config.getScreenHeight()));
    shaders.hizPyramidShader->setVec2("utexelDimensions", texelDimensions);
    
    const int downsampleWorkGroupX = config.getDownsampleWorkGroupSizeX();
    const int downsampleWorkGroupY = config.getDownsampleWorkGroupSizeY();
    
    glDispatchCompute(
        (config.getScreenWidth() + downsampleWorkGroupX - 1) / downsampleWorkGroupX,
        (config.getScreenHeight() + downsampleWorkGroupY - 1) / downsampleWorkGroupY,
        1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    
    fbo.depthTexture->unbind();
    fbo.downsampledDepthTexture->unbindImage(GL_READ_WRITE, GL_R32F);
}

/**
 * @brief Culls and renders spheres.
 * 
 * @param shaders Shader resources.
 * @param buffers Buffer resources.
 * @param camera Scene camera.
 * @param frustum View frustum.
 * @param config Render configuration.
 * @param state Mutable render state (updated with counts).
 * @param sphereCount Total sphere count.
 * @param withOcclusion Whether occlusion culling is enabled.
 */
void cullAndRenderSpheres(
    const ShaderResources& shaders,
    BufferResources& buffers,
    Camera& camera,
    Frustum& frustum,
    const RenderConfig& config,
    RenderState& state,
    int sphereCount,
    bool withOcclusion)
{
    const GLuint numGroupsX = (sphereCount + config.getWorkGroupSizeXPerSphere() - 1) 
                             / config.getWorkGroupSizeXPerSphere();
    const glm::ivec2 screenResolution(config.getScreenWidth(), config.getScreenHeight());
    
    // Culling pass
    shaders.spheresCullingShader->use();
    shaders.spheresCullingShader->setInt("sphereCount", sphereCount);
    shaders.spheresCullingShader->setVec2I("screenResolution", screenResolution);
    
    if (withOcclusion) {
        shaders.spheresCullingShader->setUint("indexOffset", 0);
    }
    
#if BENCHMARKING
    shaders.spheresCullingShader->setInt("benchmark", 1);
#else
    shaders.spheresCullingShader->setInt("benchmark", 0);
#endif
    
    setCameraUniforms(*shaders.spheresCullingShader, camera);
    setFrustumUniforms(*shaders.spheresCullingShader, frustum);
    
    glDispatchCompute(numGroupsX, 1, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
    
    // Read visibility counts
    GLuint resetValue = 0;
    
#if BENCHMARKING
    if (withOcclusion && buffers.frustumEntitiesCounter) {
        buffers.frustumEntitiesCounter->bind();
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &state.visibleSpheresCount);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
        buffers.frustumEntitiesCounter->unbind();
    }
    
    buffers.visibleEntitiesCounter->bind();
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &state.notOccludedSpheresCount);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
    buffers.visibleEntitiesCounter->unbind();
    
    if (!withOcclusion) {
        state.visibleSpheresCount = state.notOccludedSpheresCount;
    }
#else
    buffers.visibleEntitiesCounter->bind();
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &state.notOccludedSpheresCount);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
    state.visibleSpheresCount = state.notOccludedSpheresCount;
    buffers.visibleEntitiesCounter->unbind();
#endif
    
    // Render pass
    shaders.spheresShader->use();
    shaders.spheresShader->setVec2I("screenResolution", screenResolution);
    
    if (withOcclusion) {
        shaders.spheresShader->setUint("indexOffset", 0);
    }
    
    setCameraUniforms(*shaders.spheresShader, camera);
    
    glBindVertexArray(buffers.emptyVAO);
    glDrawArrays(GL_POINTS, 0, state.notOccludedSpheresCount);
}

/**
 * @brief Culls and renders cylinders.
 * 
 * @param shaders Shader resources.
 * @param buffers Buffer resources.
 * @param camera Scene camera.
 * @param frustum View frustum.
 * @param config Render configuration.
 * @param state Mutable render state (updated with counts).
 * @param cylinderCount Total cylinder count.
 * @param sphereCount Sphere count (for offset in occlusion mode).
 * @param withOcclusion Whether occlusion culling is enabled.
 */
void cullAndRenderCylinders(
    const ShaderResources& shaders,
    BufferResources& buffers,
    Camera& camera,
    Frustum& frustum,
    const RenderConfig& config,
    RenderState& state,
    int cylinderCount,
    int sphereCount,
    bool withOcclusion)
{
    const GLuint numGroupsX = (cylinderCount + config.getWorkGroupSizeXPerCylinder() - 1) 
                             / config.getWorkGroupSizeXPerCylinder();
    const glm::ivec2 screenResolution(config.getScreenWidth(), config.getScreenHeight());
    
    // Culling pass
    shaders.cylindersCullingShader->use();
    shaders.cylindersCullingShader->setInt("cylinderCount", cylinderCount);
    shaders.cylindersCullingShader->setVec2I("screenResolution", screenResolution);
    
    if (withOcclusion) {
        shaders.cylindersCullingShader->setUint("indexOffset", sphereCount);
    }
    
#if BENCHMARKING
    shaders.cylindersCullingShader->setInt("benchmark", 1);
#else
    shaders.cylindersCullingShader->setInt("benchmark", 0);
#endif
    
    setCameraUniforms(*shaders.cylindersCullingShader, camera);
    setFrustumUniforms(*shaders.cylindersCullingShader, frustum);
    
    glDispatchCompute(numGroupsX, 1, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
    
    // Read visibility counts
    GLuint resetValue = 0;
    
#if BENCHMARKING
    if (withOcclusion && buffers.frustumEntitiesCounter) {
        buffers.frustumEntitiesCounter->bind();
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &state.visibleCylindersCount);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
        buffers.frustumEntitiesCounter->unbind();
    }
    
    buffers.visibleEntitiesCounter->bind();
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &state.notOccludedCylindersCount);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
    buffers.visibleEntitiesCounter->unbind();
    
    if (!withOcclusion) {
        state.visibleCylindersCount = state.notOccludedCylindersCount;
    }
#else
    buffers.visibleEntitiesCounter->bind();
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &state.notOccludedCylindersCount);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
    state.visibleCylindersCount = state.notOccludedCylindersCount;
    buffers.visibleEntitiesCounter->unbind();
#endif
    
    // Render pass
    shaders.cylindersShader->use();
    shaders.cylindersShader->setVec2I("screenResolution", screenResolution);
    
    if (withOcclusion) {
        shaders.cylindersShader->setUint("indexOffset", sphereCount);
    }
    
    setCameraUniforms(*shaders.cylindersShader, camera);
    
    glBindVertexArray(buffers.emptyVAO);
    glDrawArrays(GL_POINTS, 0, state.notOccludedCylindersCount);
}

/**
 * @brief Performs pixel counting for occlusion feedback.
 * 
 * @param shaders Shader resources.
 * @param fbo Framebuffer resources.
 * @param config Render configuration.
 */
void performPixelCounting(
    const ShaderResources& shaders,
    const FramebufferResources& fbo,
    const RenderConfig& config)
{
    if (!shaders.pixelCountShader || !fbo.pixelIdTexture) return;
    
    fbo.pixelIdTexture->bind();
    shaders.pixelCountShader->use();
    
    const glm::ivec2 screenResolution(config.getScreenWidth(), config.getScreenHeight());
    shaders.pixelCountShader->setVec2I("screenResolution", screenResolution);
    
    glDispatchCompute(
        (config.getScreenWidth() + config.getWorkGroupSizeXPerPixel() - 1) / config.getWorkGroupSizeXPerPixel(),
        (config.getScreenHeight() + config.getWorkGroupSizeYPerPixel() - 1) / config.getWorkGroupSizeYPerPixel(),
        1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    
    fbo.pixelIdTexture->unbind();
}

/**
 * @brief Finalizes the frame by blitting to default framebuffer.
 * 
 * @param fbo Framebuffer resources.
 */
void finalizeFrame(const FramebufferResources& fbo)
{
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo.fbo->getId());
    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Bind Draw Framebuffer");
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glPopDebugGroup();
}

// ============================================================================
// Main Render Loops
// ============================================================================

/**
 * @brief Runs the render loop with occlusion culling.
 * 
 * @param camera Scene camera.
 * @param window Application window.
 * @param config Render configuration.
 * @param sceneConfig Scene configuration.
 */
void runWithOcclusionCulling(
    Camera& camera, 
    AppOpenGL& window,
    const RenderConfig& config,
    const SceneConfig& sceneConfig)
{
    CameraController cameraController(window, camera);
    
    // Create resources
    auto fbo = createFramebufferWithOcclusion(config);
    auto shaders = createShadersWithOcclusion();
    
    // Load scene data
    std::vector<Sphere> spheres = getScene(sceneConfig.getSphereCount());
    std::vector<Cylinder> cylinders = getCylinderScene(sceneConfig.getCylinderCount());
    
    int sphereCount = static_cast<int>(spheres.size());
    int cylinderCount = static_cast<int>(cylinders.size());
    
    auto checkpoints = getCheckpoints(sphereCount, spheres, cylinders);
    auto buffers = createBuffersWithOcclusion(spheres, cylinders);
    
    // Initialize render state
    RenderState state;
    state.visibleSpheresCount = sphereCount;
    state.visibleCylindersCount = cylinderCount;
    
    // Setup profiling and benchmarking
    Benchmark benchmark(cameraController, checkpoints);
    
    auto [frameTimesPath, processTimesPath] = getProfilerPaths(sceneConfig, true);
    Profiler profiler(window, frameTimesPath, processTimesPath, 
                      sphereCount, cylinderCount, 
                      config.getDownsampleLevel(), 
                      g_settings.timerDuration);
    
    // Setup UI
    auto sceneInfoMap = createSceneInfoMap(config, state, sphereCount, cylinderCount);
    window.setupSceneInfoGui("Scene Info", sceneInfoMap);
    window.setupCameraGui("Camera Info", &camera);
    window.setupInputInfoGui("General Input Info");
    window.setupBenchmarkInfoGui("Benchmark", &benchmark);
    
    // Log initialization time
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedSeconds = endTime - g_startTime;
    std::cout << "Initialization completed in " << elapsedSeconds.count() << " seconds" << std::endl;
    
    try
    {
        bool isRunning = true;
        
        while (isRunning)
        {
            // Hi-Z pyramid generation
            fbo.bind();
            performHiZPyramidGeneration(shaders, fbo, config);
            
            // Prepare frame
            prepareFrame(fbo, config);
            
            // Update camera and benchmark
            cameraController.cameraUpdate();
            benchmark.update();
            profiler.updateProfiler(
                benchmark.getCheckpointID(),
                state.visibleSpheresCount, state.notOccludedSpheresCount,
                state.visibleCylindersCount, state.notOccludedCylindersCount);
            
            if (window.getInput().isKeyDown(Key::T)) {
                profiler.startSavingNextFrames(benchmark.getCheckpointID());
            }
            
            Frustum frustum(camera);
            
            // Bind downsampled depth for occlusion queries
            glBindTextureUnit(0, fbo.downsampledDepthTexture->getId());
            
            // Render spheres and cylinders
            cullAndRenderSpheres(shaders, buffers, camera, frustum, config, 
                                state, sphereCount, true);
            cullAndRenderCylinders(shaders, buffers, camera, frustum, config, 
                                  state, cylinderCount, sphereCount, true);
            
            glBindTextureUnit(0, 0);
            
            // Pixel counting for occlusion feedback
            performPixelCounting(shaders, fbo, config);
            
            // Finalize frame
            finalizeFrame(fbo);
            isRunning = window.update();
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Render loop error: " << e.what() << std::endl;
    }
}

/**
 * @brief Runs the render loop without occlusion culling.
 * 
 * @param camera Scene camera.
 * @param window Application window.
 * @param config Render configuration.
 * @param sceneConfig Scene configuration.
 */
void runWithoutOcclusionCulling(
    Camera& camera, 
    AppOpenGL& window,
    const RenderConfig& config,
    const SceneConfig& sceneConfig)
{
    CameraController cameraController(window, camera);
    
    // Create resources
    auto fbo = createFramebufferWithoutOcclusion(config);
    auto shaders = createShadersWithoutOcclusion();
    
    // Load scene data
    std::vector<Sphere> spheres;
    std::vector<Cylinder> cylinders;
    getCompleteScene(spheres, sceneConfig.getSphereCount(), 
                     cylinders, sceneConfig.getCylinderCount());
    
    int sphereCount = static_cast<int>(spheres.size());
    int cylinderCount = static_cast<int>(cylinders.size());
    
    auto checkpoints = getCheckpoints(sphereCount, spheres, cylinders);
    auto buffers = createBuffersWithoutOcclusion(spheres, cylinders);
    
    // Clear scene data from CPU memory
    spheres.clear();
    spheres.shrink_to_fit();
    cylinders.clear();
    cylinders.shrink_to_fit();
    
    // Initialize render state
    RenderState state;
    state.visibleSpheresCount = sphereCount;
    state.visibleCylindersCount = cylinderCount;
    
    // Setup profiling and benchmarking
    Benchmark benchmark(cameraController, checkpoints);
    
    auto [frameTimesPath, processTimesPath] = getProfilerPaths(sceneConfig, false);
    Profiler profiler(window, frameTimesPath, processTimesPath, 
                      sphereCount, cylinderCount, 0, 
                      g_settings.timerDuration);
    
    // Setup UI
    auto sceneInfoMap = createSceneInfoMap(config, state, sphereCount, cylinderCount);
    window.setupSceneInfoGui("Scene Info", sceneInfoMap);
    window.setupCameraGui("Camera Info", &camera);
    window.setupInputInfoGui("General Input Info");
    window.setupBenchmarkInfoGui("Benchmark", &benchmark);
    
    // Log initialization time
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedSeconds = endTime - g_startTime;
    std::cout << "Initialization completed in " << elapsedSeconds.count() << " seconds" << std::endl;
    
    try
    {
        bool isRunning = true;
        
        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Bind Draw Framebuffer");
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glPopDebugGroup();
        
        while (isRunning)
        {
            // Prepare frame
            prepareFrame(fbo, config);
            
            // Generate mipmaps for depth
            glGenerateTextureMipmap(fbo.depthTexture->getId());
            
            // Update camera and benchmark
            cameraController.cameraUpdate();
            benchmark.update();
            profiler.updateProfiler(
                benchmark.getCheckpointID(),
                state.visibleSpheresCount, state.notOccludedSpheresCount,
                state.visibleCylindersCount, state.notOccludedCylindersCount);
            
            if (window.getInput().isKeyDown(Key::T)) {
                profiler.startSavingNextFrames(benchmark.getCheckpointID());
            }
            
            Frustum frustum(camera);
            
            // Render spheres and cylinders
            cullAndRenderSpheres(shaders, buffers, camera, frustum, config, 
                                state, sphereCount, false);
            cullAndRenderCylinders(shaders, buffers, camera, frustum, config, 
                                  state, cylinderCount, sphereCount, false);
            
            // Finalize frame
            finalizeFrame(fbo);
            isRunning = window.update();
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Render loop error: " << e.what() << std::endl;
    }
}

// ============================================================================
// Main Entry Point
// ============================================================================

/**
 * @brief Main application entry point.
 * 
 * Demonstrates the standard OpenGL rendering pipeline with:
 * - Configuration-driven initialization
 * - GPU-based frustum culling
 * - Optional Hi-Z occlusion culling
 * - Traditional vertex/fragment/geometry shader rendering
 * 
 * @param argc Argument count.
 * @param argv Argument values.
 * @return Exit code (0 for success).
 */
int main(int argc, char* argv[])
{
    std::cout << "Starting Standard OpenGL Version..." << std::endl;
    
    try
    {
        // Record start time for initialization timing
        g_startTime = std::chrono::high_resolution_clock::now();
        
        // Load configuration from file
        g_settings = SceneConfigLoader::loadDefault();
        SceneConfigLoader::applyToGlobals(g_settings);
        
        // Create configurations using Builder pattern
        RenderConfig renderConfig = createRenderConfig();
        SceneConfig sceneConfig = createSceneConfig();
        
        std::cout << "Configuration loaded successfully." << std::endl;
        std::cout << "  Screen: " << renderConfig.getScreenWidth() << "x" 
                  << renderConfig.getScreenHeight() << std::endl;
        std::cout << "  Occlusion culling: " 
                  << (renderConfig.hasOcclusionCulling() ? "enabled" : "disabled") 
                  << std::endl;
        
        // Create window and camera
        AppOpenGL window(renderConfig.getTitle(), 
                        renderConfig.getScreenWidth(), 
                        renderConfig.getScreenHeight(), 
                        renderConfig.isShown());
        
        Camera camera(renderConfig.getScreenWidth(), renderConfig.getScreenHeight());
        camera.SetPosition(0.0f, 0.0f, 0.0f);
        
        // Run appropriate render loop based on configuration
        if (renderConfig.hasOcclusionCulling())
        {
            runWithOcclusionCulling(camera, window, renderConfig, sceneConfig);
        }
        else
        {
            runWithoutOcclusionCulling(camera, window, renderConfig, sceneConfig);
        }
        
        std::cout << "Application finished successfully." << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
