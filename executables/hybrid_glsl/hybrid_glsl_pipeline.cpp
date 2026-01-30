/**
 * @file hybrid_glsl_pipeline.cpp
 * @brief Hybrid GLSL Rasterization Pipeline
 * 
 * ============================================================================
 * OVERVIEW
 * ============================================================================
 * 
 * This executable implements a hybrid rasterization approach using OpenGL
 * compute shaders. It combines two strategies:
 * 
 *   1. SMALL ENTITIES (1st version approach):
 *      - Direct rasterization with 1 thread per entity
 *      - Each thread iterates over its bbox pixels
 *      - Efficient when entities cover few pixels
 * 
 *   2. LARGE ENTITIES (2nd version approach):
 *      - Tiled rasterization with 1 thread per pixel
 *      - Each tile processes its assigned entities
 *      - Entities are recomputed in shared memory (no global billboard buffer)
 *      - Efficient when entities cover many pixels
 * 
 * ============================================================================
 * PIPELINE STAGES
 * ============================================================================
 * 
 *   Stage 0: screen_clear.compute
 *            - Clear color buffer to black
 *            - Reset depth buffer to far plane
 *            - Reset ALL counters (tile + small entity)
 * 
 *   Stage 1: sphere_classify.compute
 *            - Frustum culling
 *            - BBox extraction
 *            - If small: add index to small sphere list
 *            - If large: add index to overlapping tiles
 * 
 *   Stage 2: cylinder_classify.compute
 *            - Same as Stage 1 for cylinders
 * 
 *   Stage 3: small_sphere_raster.compute
 *            - Direct rasterization for small spheres
 *            - One thread per sphere
 * 
 *   Stage 4: small_cylinder_raster.compute
 *            - Direct rasterization for small cylinders
 *            - One thread per cylinder
 * 
 *   Stage 5: tiled_sphere_raster.compute
 *            - Tiled rasterization for large spheres
 *            - Each tile recomputes bbox in shared memory
 * 
 *   Stage 6: tiled_cylinder_raster.compute
 *            - Tiled rasterization for large cylinders
 *            - Each tile recomputes bbox in shared memory
 * 
 * ============================================================================
 * KEY DESIGN DECISIONS
 * ============================================================================
 * 
 *   - Classification is SEPARATE from rasterization
 *     Cleaner code, better separation of concerns
 * 
 *   - No global billboard buffer
 *     Each tile recomputes bbox from original entity data
 *     Trades compute for memory bandwidth
 * 
 *   - Fixed tile size: 256 entities max per tile
 *     Simple and predictable, overflow is ignored
 * 
 *   - Shared memory for batching in tiled shader
 *     64 entities per batch, cooperatively loaded
 * 
 * ============================================================================
 * ARCHITECTURE
 * ============================================================================
 * 
 *   - HybridGLSLConfig: Configuration data structure
 *   - HybridGLSLResources: GPU buffer management
 *   - HybridGLSLPipeline: Shader orchestration
 * 
 * ============================================================================
 */

#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <string>
#include <chrono>
#include <unordered_map>

// GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// OpenGL and SDL
#include <glad/glad.h>
#include <SDL3/SDL.h>

// Project includes
#include "appSDLGL.h"
#include "geometry/cylinder/cylinder.h"
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

using uint = unsigned int;

// ============================================================================
// CONFIGURATION
// ============================================================================

/**
 * @brief Configuration for the hybrid GLSL pipeline
 */
struct HybridGLSLConfig {
    // -------------------------------------------------------------------------
    // Screen dimensions
    // -------------------------------------------------------------------------
    int screenWidth = 1024;
    int screenHeight = 1024;
    
    // -------------------------------------------------------------------------
    // Entity counts (loaded from scene)
    // -------------------------------------------------------------------------
    int sphereCount = 0;
    int cylinderCount = 0;
    
    // -------------------------------------------------------------------------
    // Classification threshold
    // -------------------------------------------------------------------------
    // Entities with bbox area <= this value use direct rasterization
    // Entities with bbox area > this value use tiled rasterization
    int smallEntityThreshold = 256;  // 16x16 pixels
    
    // -------------------------------------------------------------------------
    // Tile configuration (fixed values, displayed in GUI)
    // -------------------------------------------------------------------------
    int tileSize = 16;
    int maxEntitiesPerTile = 256;

    // -------------------------------------------------------------------------
    // Ray Casting Structure
    // -------------------------------------------------------------------------
    glm::vec3 rayStart;
    glm::vec3 rayDx;
    glm::vec3 rayDy;

    // -------------------------------------------------------------------------
    // Derived values (computed from above)
    // -------------------------------------------------------------------------
    int tilesX;
    int tilesY;
    int totalTiles;
    
    // -------------------------------------------------------------------------
    // Benchmark settings
    // -------------------------------------------------------------------------
    double timerDuration = 48.0;
    bool shown = true;
    
    /**
     * @brief Compute derived configuration values
     */
    void computeDerivedValues() {
        tilesX = (screenWidth + tileSize - 1) / tileSize;
        tilesY = (screenHeight + tileSize - 1) / tileSize;
        totalTiles = tilesX * tilesY;
    }
    
    /**
     * @brief Print configuration summary
     */
    void print() const {
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Screen: " << screenWidth << "x" << screenHeight << std::endl;
        std::cout << "  Tiles: " << tilesX << "x" << tilesY 
                  << " (" << totalTiles << " total)" << std::endl;
        std::cout << "  Small entity threshold: " << smallEntityThreshold << " pixels^2" << std::endl;
        std::cout << "  Max entities per tile: " << maxEntitiesPerTile << std::endl;
    }
};

// ============================================================================
// RESOURCE MANAGER
// ============================================================================

/**
 * @brief Manages GPU resources for the hybrid pipeline
 */
class HybridGLSLResources {
public:
    HybridGLSLResources() = default;
    ~HybridGLSLResources() = default;
    
    /**
     * @brief Initialize all GPU buffers
     */
    void initialize(const HybridGLSLConfig& config,
                    const std::vector<Sphere>& spheres,
                    const std::vector<Cylinder>& cylinders) {
        m_config = config;
        
        std::cout << "Initializing GPU resources..." << std::endl;
        
        // ---------------------------------------------------------------------
        // Binding 0: Output image (handled by Canvas)
        // ---------------------------------------------------------------------
        
        // ---------------------------------------------------------------------
        // Binding 1: Depth Buffer
        // ---------------------------------------------------------------------
        m_depthBuffer = std::make_unique<StorageBuffer>(
            GL_SHADER_STORAGE_BUFFER,
            config.screenWidth * config.screenHeight * sizeof(GLuint),
            1, nullptr, GL_DYNAMIC_COPY);
        
        // ---------------------------------------------------------------------
        // Binding 2-3: Tile Counters
        // ---------------------------------------------------------------------
        m_tileSphereCounts = std::make_unique<StorageBuffer>(
            GL_SHADER_STORAGE_BUFFER,
            config.totalTiles * sizeof(GLuint),
            2, nullptr, GL_DYNAMIC_COPY);
        
        m_tileCylinderCounts = std::make_unique<StorageBuffer>(
            GL_SHADER_STORAGE_BUFFER,
            config.totalTiles * sizeof(GLuint),
            3, nullptr, GL_DYNAMIC_COPY);
        
        // ---------------------------------------------------------------------
        // Binding 4-5: Tile Entity Indices
        // ---------------------------------------------------------------------
        size_t tileIndicesSize = config.totalTiles * config.maxEntitiesPerTile * sizeof(GLuint);
        
        m_tileSphereIndices = std::make_unique<StorageBuffer>(
            GL_SHADER_STORAGE_BUFFER,
            tileIndicesSize,
            4, nullptr, GL_DYNAMIC_COPY);
        
        m_tileCylinderIndices = std::make_unique<StorageBuffer>(
            GL_SHADER_STORAGE_BUFFER,
            tileIndicesSize,
            5, nullptr, GL_DYNAMIC_COPY);
        
        // ---------------------------------------------------------------------
        // Binding 6-7: Entity Buffers (input)
        // ---------------------------------------------------------------------
        m_sphereBuffer = std::make_unique<StorageBuffer>(
            GL_SHADER_STORAGE_BUFFER,
            config.sphereCount * sizeof(Sphere),
            6, spheres.data(), GL_STATIC_DRAW);
        
        m_cylinderBuffer = std::make_unique<StorageBuffer>(
            GL_SHADER_STORAGE_BUFFER,
            config.cylinderCount * sizeof(Cylinder),
            7, cylinders.data(), GL_STATIC_DRAW);
        
        // ---------------------------------------------------------------------
        // Binding 10-11: Small Entity Counters
        // ---------------------------------------------------------------------
        GLuint zero = 0;
        m_smallSphereCount = std::make_unique<StorageBuffer>(
            GL_SHADER_STORAGE_BUFFER,
            sizeof(GLuint),
            10, &zero, GL_DYNAMIC_COPY);
        
        m_smallCylinderCount = std::make_unique<StorageBuffer>(
            GL_SHADER_STORAGE_BUFFER,
            sizeof(GLuint),
            11, &zero, GL_DYNAMIC_COPY);
        
        // ---------------------------------------------------------------------
        // Binding 12-13: Small Entity Index Lists
        // ---------------------------------------------------------------------
        // Worst case: all entities are small
        m_smallSphereIndices = std::make_unique<StorageBuffer>(
            GL_SHADER_STORAGE_BUFFER,
            config.sphereCount * sizeof(GLuint),
            12, nullptr, GL_DYNAMIC_COPY);
        
        m_smallCylinderIndices = std::make_unique<StorageBuffer>(
            GL_SHADER_STORAGE_BUFFER,
            config.cylinderCount * sizeof(GLuint),
            13, nullptr, GL_DYNAMIC_COPY);
        
        // Print memory usage
        size_t totalMem = 
            config.screenWidth * config.screenHeight * sizeof(GLuint) +  // depth
            config.totalTiles * sizeof(GLuint) * 2 +                     // tile counters
            tileIndicesSize * 2 +                                         // tile indices
            config.sphereCount * sizeof(Sphere) +                        // spheres
            config.cylinderCount * sizeof(Cylinder) +                    // cylinders
            sizeof(GLuint) * 2 +                                          // small counters
            config.sphereCount * sizeof(GLuint) +                        // small sphere indices
            config.cylinderCount * sizeof(GLuint);                       // small cylinder indices
        
        std::cout << "  Total GPU memory: " << (totalMem / 1024 / 1024) << " MB" << std::endl;
    }
    
    /**
     * @brief Get small sphere count (for dispatch size)
     */
    GLuint getSmallSphereCount() {
        GLuint count = 0;
        m_smallSphereCount->bind();
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &count);
        return count;
    }
    
    /**
     * @brief Get small cylinder count (for dispatch size)
     */
    GLuint getSmallCylinderCount() {
        GLuint count = 0;
        m_smallCylinderCount->bind();
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &count);
        return count;
    }
    
private:
    HybridGLSLConfig m_config;
    
    // Core buffers
    std::unique_ptr<StorageBuffer> m_depthBuffer;
    
    // Tile data
    std::unique_ptr<StorageBuffer> m_tileSphereCounts;
    std::unique_ptr<StorageBuffer> m_tileCylinderCounts;
    std::unique_ptr<StorageBuffer> m_tileSphereIndices;
    std::unique_ptr<StorageBuffer> m_tileCylinderIndices;
    
    // Entity data
    std::unique_ptr<StorageBuffer> m_sphereBuffer;
    std::unique_ptr<StorageBuffer> m_cylinderBuffer;
    
    // Small entity data
    std::unique_ptr<StorageBuffer> m_smallSphereCount;
    std::unique_ptr<StorageBuffer> m_smallCylinderCount;
    std::unique_ptr<StorageBuffer> m_smallSphereIndices;
    std::unique_ptr<StorageBuffer> m_smallCylinderIndices;
};

// ============================================================================
// PIPELINE EXECUTOR
// ============================================================================

/**
 * @brief Executes the hybrid GLSL pipeline
 */
class HybridGLSLPipeline {
public:
    explicit HybridGLSLPipeline(const HybridGLSLConfig& config)
        : m_config(config) {
    }
    
    /**
     * @brief Initialize shaders
     */
    void initialize() {
        std::cout << "Loading compute shaders..." << std::endl;
        
        // Stage 0: Screen clearing + counter reset
        m_screenClearShader = std::make_unique<ComputeShader>(
            "assets/shaders/hybrid_glsl/screen_clear.compute",
            "Screen Clear");
        
        // Stage 1: Classification (separate for spheres and cylinders)
        m_sphereClassifyShader = std::make_unique<ComputeShader>(
            "assets/shaders/hybrid_glsl/sphere_classify.compute",
            "Sphere Classify");
        
        // Stage 3: Small entity direct rasterization
        m_smallSphereRasterShader = std::make_unique<ComputeShader>(
            "assets/shaders/hybrid_glsl/small_sphere_raster.compute",
            "Small Sphere Raster");

        // Stage 5: Tiled rasterization for large entities (separate shaders)
        m_tiledSphereRasterShader = std::make_unique<ComputeShader>(
            "assets/shaders/hybrid_glsl/tiled_sphere_raster.compute",
            "Tiled Sphere Raster");

        m_cylinderClassifyShader = std::make_unique<ComputeShader>(
            "assets/shaders/hybrid_glsl/cylinder_classify.compute",
            "Cylinder Classify");
        
        m_smallCylinderRasterShader = std::make_unique<ComputeShader>(
            "assets/shaders/hybrid_glsl/small_cylinder_raster.compute",
            "Small Cylinder Raster");
        
        m_tiledCylinderRasterShader = std::make_unique<ComputeShader>(
            "assets/shaders/hybrid_glsl/tiled_cylinder_raster.compute",
            "Tiled Cylinder Raster");
        
        std::cout << "  All shaders loaded successfully." << std::endl;
    }

    void computeScreenRayCasting(Camera& camera) 
    {
        // Compute screen ray casting parameters
        float aspectRatio = static_cast<float>(m_config.screenWidth) / 
        static_cast<float>(m_config.screenHeight);
        float fovRad = glm::radians(camera.getFov());
        float fovTan = std::tan(fovRad * 0.5f);
        float halfFovTan = fovTan * aspectRatio;

        glm::vec3 front = camera.getFront();
        glm::vec3 up = camera.getUp();
        glm::vec3 right = camera.getRight();

        // View-space ray directions for screen corners
        glm::vec3 corner00 = glm::normalize((-1.0f * halfFovTan) * right + 
                            (-1.0f * fovTan) * up + front);
        glm::vec3 corner10 = glm::normalize(( 1.0f * halfFovTan) * right + 
                            (-1.0f * fovTan) * up + front);
        glm::vec3 corner01 = glm::normalize((-1.0f * halfFovTan) * right + 
                            ( 1.0f * fovTan) * up + front);

        // Transform to world space
        glm::mat3 viewMat3(camera.getView());
        glm::vec3 wCorner00 = viewMat3 * corner00;
        glm::vec3 wCorner10 = viewMat3 * corner10;
        glm::vec3 wCorner01 = viewMat3 * corner01;

        // Delta per pixel
        m_config.rayDx = (wCorner10 - wCorner00) / static_cast<float>(m_config.screenWidth);
        m_config.rayDy = (wCorner01 - wCorner00) / static_cast<float>(m_config.screenHeight);
        m_config.rayStart = wCorner00;
    }
    
    /**
     * @brief Execute one frame of the pipeline
     * @param config Reference to config for runtime-adjustable parameters
     */
    void execute(Camera& camera, const Frustum& frustum,
                 HybridGLSLResources& resources,
                 HybridGLSLConfig& config) {
        
        // Update runtime-adjustable parameters from external config
        m_config.smallEntityThreshold = config.smallEntityThreshold;
        m_config.maxEntitiesPerTile = config.maxEntitiesPerTile;
        
        glm::ivec2 screenRes(m_config.screenWidth, m_config.screenHeight);
        
        computeScreenRayCasting(camera);

        // =====================================================================
        // STAGE 0: Screen Clear + Reset ALL Counters
        // =====================================================================
        
        m_screenClearShader->use();
        m_screenClearShader->setVec2I("screenResolution", screenRes);
        m_screenClearShader->setFloat("farPlane", camera.getFar());
        m_screenClearShader->setInt("tilesX", m_config.tilesX);
        m_screenClearShader->setInt("tilesY", m_config.tilesY);
        
        glDispatchCompute(
            (m_config.screenWidth + 15) / 16,
            (m_config.screenHeight + 15) / 16,
            1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        
        // =====================================================================
        // STAGE 1: Sphere Classification
        // =====================================================================
        // Classifies each sphere as small or large
        // Small: index added to small sphere list
        // Large: index added to overlapping tiles
        
        m_sphereClassifyShader->use();
        setClassifyUniforms(*m_sphereClassifyShader, camera, frustum);
        m_sphereClassifyShader->setInt("sphereCount", m_config.sphereCount);
        m_sphereClassifyShader->setInt("smallEntityThreshold", m_config.smallEntityThreshold);
        m_sphereClassifyShader->setInt("tilesX", m_config.tilesX);
        m_sphereClassifyShader->setInt("tilesY", m_config.tilesY);
        
        glDispatchCompute((m_config.sphereCount + 255) / 256, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        
        
        // =====================================================================
        // STAGE 2: Small Sphere Direct Rasterization
        // =====================================================================
        // One thread per small sphere, iterates over bbox pixels
        
        // Get count for dispatch (requires sync)
        GLuint smallSphereCount = resources.getSmallSphereCount();
        
        if (smallSphereCount > 0) {
            m_smallSphereRasterShader->use();
            setRasterUniforms(*m_smallSphereRasterShader, camera);
            
            glDispatchCompute((smallSphereCount + 255) / 256, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        }
        
        // =====================================================================
        // STAGE 3: Tiled Rasterization for Large Spheres
        // =====================================================================
        // Each tile processes its assigned large spheres
        // BBox is recomputed in shared memory (no global billboard buffer)
        
        m_tiledSphereRasterShader->use();
        setRasterUniforms(*m_tiledSphereRasterShader, camera);
        m_tiledSphereRasterShader->setInt("tilesX", m_config.tilesX);
        m_tiledSphereRasterShader->setInt("tilesY", m_config.tilesY);
        
        glDispatchCompute(m_config.tilesX, m_config.tilesY, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);


        // =====================================================================
        // STAGE 4: Cylinder Classification
        // =====================================================================
        
        m_cylinderClassifyShader->use();
        setClassifyUniforms(*m_cylinderClassifyShader, camera, frustum);
        m_cylinderClassifyShader->setInt("cylinderCount", m_config.cylinderCount);
        m_cylinderClassifyShader->setInt("smallEntityThreshold", m_config.smallEntityThreshold);
        m_cylinderClassifyShader->setInt("tilesX", m_config.tilesX);
        m_cylinderClassifyShader->setInt("tilesY", m_config.tilesY);
        
        glDispatchCompute((m_config.cylinderCount + 255) / 256, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        
        // =====================================================================
        // STAGE 5: Small Cylinder Direct Rasterization
        // =====================================================================
        
        GLuint smallCylinderCount = resources.getSmallCylinderCount();
        
        if (smallCylinderCount > 0) {
            m_smallCylinderRasterShader->use();
            setRasterUniforms(*m_smallCylinderRasterShader, camera);
            
            glDispatchCompute((smallCylinderCount + 255) / 256, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        }
        
        
        // =====================================================================
        // STAGE 6: Tiled Rasterization for Large Cylinders
        // =====================================================================
        // Each tile processes its assigned large cylinders
        
        m_tiledCylinderRasterShader->use();
        setRasterUniforms(*m_tiledCylinderRasterShader, camera);
        m_tiledCylinderRasterShader->setInt("tilesX", m_config.tilesX);
        m_tiledCylinderRasterShader->setInt("tilesY", m_config.tilesY);
        m_tiledCylinderRasterShader->setInt("maxEntitiesPerTile", m_config.maxEntitiesPerTile);
        
        glDispatchCompute(m_config.tilesX, m_config.tilesY, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }
    
private:
    /**
     * @brief Set uniforms for classification shaders
     */
    void setClassifyUniforms(ComputeShader& shader, Camera& camera, 
                              const Frustum& frustum) {
        glm::ivec2 screenRes(m_config.screenWidth, m_config.screenHeight);
        
        shader.setMat4("view", camera.getView());
        shader.setMat4("proj", camera.getProjection());
        
        shader.setVec4("frustumTopFace", 
            glm::vec4(frustum.topFace.normal, frustum.topFace.distance));
        shader.setVec4("frustumBottomFace", 
            glm::vec4(frustum.bottomFace.normal, frustum.bottomFace.distance));
        shader.setVec4("frustumRightFace", 
            glm::vec4(frustum.rightFace.normal, frustum.rightFace.distance));
        shader.setVec4("frustumLeftFace", 
            glm::vec4(frustum.leftFace.normal, frustum.leftFace.distance));
        shader.setVec4("frustumFarFace", 
            glm::vec4(frustum.farFace.normal, frustum.farFace.distance));
        shader.setVec4("frustumNearFace", 
            glm::vec4(frustum.nearFace.normal, frustum.nearFace.distance));
        
        shader.setVec2I("screenResolution", screenRes);
    }
    
    /**
     * @brief Set uniforms for rasterization shaders
     */
    void setRasterUniforms(ComputeShader& shader, Camera& camera) {
        glm::ivec2 screenRes(m_config.screenWidth, m_config.screenHeight);
        
        shader.setMat4("view", camera.getView());
        shader.setMat4("proj", camera.getProjection());
        
        shader.setVec3("front", camera.getFront());
        shader.setVec3("up", camera.getUp());
        shader.setVec3("right", camera.getRight());
        shader.setVec3("cameraPos", camera.getPosition());
        
        shader.setVec2I("screenResolution", screenRes);
        shader.setFloat("fov", camera.getFov());
        
        
        shader.setVec3("rayStart", m_config.rayStart);
        shader.setVec3("rayDx", m_config.rayDx);
        shader.setVec3("rayDy", m_config.rayDy);
    }
    
    HybridGLSLConfig m_config;
    
    // Shaders
    std::unique_ptr<ComputeShader> m_screenClearShader;
    std::unique_ptr<ComputeShader> m_sphereClassifyShader;
    std::unique_ptr<ComputeShader> m_cylinderClassifyShader;
    std::unique_ptr<ComputeShader> m_smallSphereRasterShader;
    std::unique_ptr<ComputeShader> m_smallCylinderRasterShader;
    std::unique_ptr<ComputeShader> m_tiledSphereRasterShader;
    std::unique_ptr<ComputeShader> m_tiledCylinderRasterShader;
};

// ============================================================================
// CONFIGURATION LOADER
// ============================================================================

HybridGLSLConfig loadConfiguration() {
    HybridGLSLConfig config;
    
    SceneSettings settings = SceneConfigLoader::loadDefault();
    SceneConfigLoader::applyToGlobals(settings);
    
    config.screenWidth = settings.screenWidth;
    config.screenHeight = settings.screenHeight;
    config.sphereCount = settings.sphereCount;
    config.cylinderCount = settings.cylinderCount;
    config.timerDuration = settings.timerDuration;
    config.smallEntityThreshold = 256;
    
    config.computeDerivedValues();
    
    return config;
}


// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char* argv[]) {
    // =========================================================================
    // Banner
    // =========================================================================
    
    std::cout << "============================================================" << std::endl;
    std::cout << "  Hybrid GLSL Rasterization Pipeline                        " << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << std::endl;
    std::cout << "Pipeline:" << std::endl;
    std::cout << "  1. Classify entities (small vs large)" << std::endl;
    std::cout << "  2. Small: Direct rasterization (1 thread/entity)" << std::endl;
    std::cout << "  3. Large: Tiled rasterization (1 thread/pixel)" << std::endl;
    std::cout << "  4. No global billboard buffer (recompute in shared mem)" << std::endl;
    std::cout << std::endl;
    
    // =========================================================================
    // Load Configuration
    // =========================================================================
    
    HybridGLSLConfig config = loadConfiguration();
    config.print();
    std::cout << std::endl;
    
    // =========================================================================
    // Create Window
    // =========================================================================
    
    std::string title = "Hybrid GLSL Pipeline";
    AppOpenGL window(title, config.screenWidth, config.screenHeight, config.shown);
    
    // =========================================================================
    // Create Camera
    // =========================================================================
    
    Camera camera(config.screenWidth, config.screenHeight);
    camera.SetPosition(0.0f, 0.0f, 0.0f);
    
    CameraController cameraController(window, camera);
    
    // =========================================================================
    // Load Scene
    // =========================================================================
    
    std::cout << "Loading scene..." << std::endl;
    
    std::vector<Sphere> spheres;
    std::vector<Cylinder> cylinders;
    getCompleteScene(spheres, config.sphereCount, cylinders, config.cylinderCount);
    
    config.sphereCount = static_cast<int>(spheres.size());
    config.cylinderCount = static_cast<int>(cylinders.size());
    
    std::cout << "  Spheres: " << config.sphereCount << std::endl;
    std::cout << "  Cylinders: " << config.cylinderCount << std::endl;
    std::cout << std::endl;
    
    config.print();
    std::cout << std::endl;
    
    // Get checkpoints for benchmark
    std::vector<std::pair<glm::vec3, glm::vec3>> checkpoints = 
        getCheckpoints(config.sphereCount, spheres, cylinders);
    
    // =========================================================================
    // Create Canvas
    // =========================================================================
    
    Canvas canvas(GL_TEXTURE_2D, GL_RGBA8, config.screenWidth, config.screenHeight);
    canvas.setFBO(GL_COLOR_ATTACHMENT0);
    canvas.bindTexture();
    canvas.bindFBO();
    
    // =========================================================================
    // Initialize Resources
    // =========================================================================
    
    HybridGLSLResources resources;
    resources.initialize(config, spheres, cylinders);
    
    spheres.clear();
    cylinders.clear();
    
    // =========================================================================
    // Initialize Pipeline
    // =========================================================================
    
    HybridGLSLPipeline pipeline(config);
    pipeline.initialize();
    
    // =========================================================================
    // Setup Benchmark
    // =========================================================================
    
    Benchmark benchmark(cameraController, checkpoints);
    
    std::string basePath = "media/csv/hybrid_glsl/";
    std::string frameTimesPath = basePath + "frame_times.csv";
    std::string processTimesPath = basePath + "process_times.csv";
    
    int visibleSpheres = config.sphereCount;
    int drawnSpheres = config.sphereCount;
    int visibleCylinders = config.cylinderCount;
    int drawnCylinders = config.cylinderCount;
    
    Profiler profiler(window, frameTimesPath, processTimesPath,
                      config.sphereCount, config.cylinderCount, 0, config.timerDuration);
    
    // =========================================================================
    // Setup GUI
    // =========================================================================
    
    std::unordered_map<std::string, int*> sceneData = {
        {"Screen width", &config.screenWidth},
        {"Screen height", &config.screenHeight},
        {"Sphere count", &config.sphereCount},
        {"Visible spheres", &visibleSpheres},
        {"Drawn spheres", &drawnSpheres},
        {"Cylinder count", &config.cylinderCount},
        {"Visible cylinders", &visibleCylinders},
        {"Drawn cylinders", &drawnCylinders}
    };
    
    window.setupSceneInfoGui("Scene Info", sceneData);
    window.setupCameraGui("Camera Info", &camera);
    window.setupInputInfoGui("Input Info");
    window.setupBenchmarkInfoGui("Benchmark", &benchmark);
    window.setupPipelineConfigGui("Pipeline Config", 
                                  config.smallEntityThreshold,
                                  config.tileSize,
                                  config.maxEntitiesPerTile,
                                  config.tilesX,
                                  config.tilesY);
    
    // =========================================================================
    // Main Loop
    // =========================================================================
    
    std::cout << "Starting render loop..." << std::endl;
    std::cout << "  F10: Screenshot" << std::endl;
    std::cout << "  T: Start profiling" << std::endl;
    std::cout << std::endl;
    
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    bool isRunning = true;
    
    while (isRunning) {
        cameraController.cameraUpdate();
        benchmark.update();
        profiler.updateProfiler(benchmark.getCheckpointID(),
                                visibleSpheres, drawnSpheres,
                                visibleCylinders, drawnCylinders);
        
        if (window.getInput().isKeyDown(Key::T)) {
            profiler.startSavingNextFrames(benchmark.getCheckpointID());
        }
        
        Frustum frustum(camera);
        
        pipeline.execute(camera, frustum, resources, config);
        
        glBindFramebuffer(GL_READ_FRAMEBUFFER, canvas.getFramebuffer().getId());
        isRunning = window.update();
        
        if (window.getInput().isKeyDown(Key::F10)) {
            std::string sshotName = "media/img/hybrid_glsl/frame_" + 
                                    std::to_string(benchmark.getCheckpointID()) + ".bmp";
            canvas.takeScreenshot(sshotName);
            std::cout << "Screenshot: " << sshotName << std::endl;
        }
    }
    
    // =========================================================================
    // Cleanup
    // =========================================================================
    
    std::cout << std::endl;
    std::cout << "Done." << std::endl;
    
    return 0;
}
