// ============================================================================
// hybrid_glsl_pipeline.cpp - Hybrid GLSL Rasterization Pipeline
// ============================================================================
//
// This file implements a hybrid molecular rasterization pipeline using
// OpenGL Compute Shaders. The pipeline combines two approaches:
//
// 1. DIRECT RASTERIZATION (for small entities):
//    - One thread per entity
//    - Each thread iterates over its bbox pixels
//    - Efficient for entities with small screen footprint
//
// 2. TILED RASTERIZATION (for large entities):
//    - Screen divided into 16x16 pixel tiles
//    - Each tile maintains a list of overlapping entities
//    - One thread per pixel, processes all entities in tile
//    - Efficient for entities with large screen footprint
//
// Pipeline Stages:
//   Stage 0: Screen Clear (reset buffers and counters)
//   Stage 1: Sphere Classification (small vs large, assigns to lists/tiles)
//   Stage 2: Small Sphere Direct Rasterization
//   Stage 3: Large Sphere Tiled Rasterization
//   Stage 4: Cylinder Classification
//   Stage 5: Small Cylinder Direct Rasterization
//   Stage 6: Large Cylinder Tiled Rasterization
//
// ============================================================================

#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <cmath>

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "appSDLGL.h"
#include "ux/camera.h"
#include "ux/camera_controller.h"
#include "ux/input.h"
#include "ux/cinematic/benchmark.h"
#include "ux/profiler/profiler.h"
#include "geometry/cylinder/cylinder.h"
#include "algorithms/frustum_cull.h"
#include "utils/benchmark_resources.h"

#include "vis/gl/storage_buffer.h"
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
                    const std::vector<glm::vec4>& spheres,
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
        // Binding 6: Sphere Data (already in glm::vec4 format: xyz=position, w=radius)
        // ---------------------------------------------------------------------
        m_sphereBuffer = std::make_unique<StorageBuffer>(
            GL_SHADER_STORAGE_BUFFER,
            spheres.size() * sizeof(glm::vec4),
            6, spheres.data(), GL_STATIC_DRAW);
        
        // ---------------------------------------------------------------------
        // Binding 7: Cylinder Data (CylinderIndex struct with sphereIndexA, sphereIndexB, radius)
        // ---------------------------------------------------------------------
        m_cylinderBuffer = std::make_unique<StorageBuffer>(
            GL_SHADER_STORAGE_BUFFER,
            cylinders.size() * sizeof(Cylinder),
            7, cylinders.data(), GL_STATIC_DRAW);
        
        // ---------------------------------------------------------------------
        // Binding 10-11: Small Entity Counters
        // ---------------------------------------------------------------------
        m_smallSphereCounter = std::make_unique<StorageBuffer>(
            GL_SHADER_STORAGE_BUFFER,
            sizeof(GLuint),
            10, nullptr, GL_DYNAMIC_COPY);
        
        m_smallCylinderCounter = std::make_unique<StorageBuffer>(
            GL_SHADER_STORAGE_BUFFER,
            sizeof(GLuint),
            11, nullptr, GL_DYNAMIC_COPY);
        
        // ---------------------------------------------------------------------
        // Binding 12-13: Small Entity Index Lists
        // ---------------------------------------------------------------------
        m_smallSphereIndices = std::make_unique<StorageBuffer>(
            GL_SHADER_STORAGE_BUFFER,
            config.sphereCount * sizeof(GLuint),
            12, nullptr, GL_DYNAMIC_COPY);
        
        m_smallCylinderIndices = std::make_unique<StorageBuffer>(
            GL_SHADER_STORAGE_BUFFER,
            config.cylinderCount * sizeof(GLuint),
            13, nullptr, GL_DYNAMIC_COPY);
        
        std::cout << "  Depth buffer: " 
                  << (config.screenWidth * config.screenHeight * sizeof(GLuint)) / 1024 
                  << " KB" << std::endl;
        std::cout << "  Tile indices: " 
                  << (2 * tileIndicesSize) / (1024 * 1024) 
                  << " MB" << std::endl;
        std::cout << "  Sphere data: " 
                  << (spheres.size() * sizeof(glm::vec4)) / 1024 
                  << " KB" << std::endl;
        std::cout << "  Cylinder data: " 
                  << (cylinders.size() * sizeof(Cylinder)) / 1024 
                  << " KB" << std::endl;
    }
    
    /**
     * @brief Get small sphere count from GPU (requires sync)
     */
    GLuint getSmallSphereCount() {
        GLuint count = 0;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_smallSphereCounter->getId());
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &count);
        return count;
    }
    
    /**
     * @brief Get small cylinder count from GPU (requires sync)
     */
    GLuint getSmallCylinderCount() {
        GLuint count = 0;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_smallCylinderCounter->getId());
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &count);
        return count;
    }
    
private:
    HybridGLSLConfig m_config;
    
    // Output and depth
    std::unique_ptr<StorageBuffer> m_depthBuffer;
    
    // Tile management
    std::unique_ptr<StorageBuffer> m_tileSphereCounts;
    std::unique_ptr<StorageBuffer> m_tileCylinderCounts;
    std::unique_ptr<StorageBuffer> m_tileSphereIndices;
    std::unique_ptr<StorageBuffer> m_tileCylinderIndices;
    
    // Entity data
    std::unique_ptr<StorageBuffer> m_sphereBuffer;
    std::unique_ptr<StorageBuffer> m_cylinderBuffer;
    
    // Small entity tracking
    std::unique_ptr<StorageBuffer> m_smallSphereCounter;
    std::unique_ptr<StorageBuffer> m_smallCylinderCounter;
    std::unique_ptr<StorageBuffer> m_smallSphereIndices;
    std::unique_ptr<StorageBuffer> m_smallCylinderIndices;
};

// ============================================================================
// PIPELINE
// ============================================================================

/**
 * @brief The hybrid GLSL rasterization pipeline
 */
class HybridGLSLPipeline {
public:
    HybridGLSLPipeline() = default;
    ~HybridGLSLPipeline() = default;
    
    /**
     * @brief Initialize the pipeline with configuration
     */
    void initialize(const HybridGLSLConfig& config) {
        m_config = config;
        
        std::cout << "Loading shaders..." << std::endl;
        
        // Stage 0: Screen clear
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
    
    /**
     * @brief Execute one frame of the pipeline
     */
    void execute(Camera& camera, const Frustum& frustum,
                 HybridGLSLResources& resources) {
        
        glm::ivec2 screenRes(m_config.screenWidth, m_config.screenHeight);
        
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


        // // =====================================================================
        // // STAGE 4: Cylinder Classification
        // // =====================================================================
        
        // m_cylinderClassifyShader->use();
        // setClassifyUniforms(*m_cylinderClassifyShader, camera, frustum);
        // m_cylinderClassifyShader->setInt("cylinderCount", m_config.cylinderCount);
        // m_cylinderClassifyShader->setInt("smallEntityThreshold", m_config.smallEntityThreshold);
        // m_cylinderClassifyShader->setInt("tilesX", m_config.tilesX);
        // m_cylinderClassifyShader->setInt("tilesY", m_config.tilesY);
        
        // glDispatchCompute((m_config.cylinderCount + 255) / 256, 1, 1);
        // glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        
        // // =====================================================================
        // // STAGE 5: Small Cylinder Direct Rasterization
        // // =====================================================================
        
        // GLuint smallCylinderCount = resources.getSmallCylinderCount();
        
        // if (smallCylinderCount > 0) {
        //     m_smallCylinderRasterShader->use();
        //     setRasterUniforms(*m_smallCylinderRasterShader, camera);
            
        //     glDispatchCompute((smallCylinderCount + 255) / 256, 1, 1);
        //     glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        // }
        
        
        // // =====================================================================
        // // STAGE 6: Tiled Rasterization for Large Cylinders
        // // =====================================================================
        // // Each tile processes its assigned large cylinders
        
        // m_tiledCylinderRasterShader->use();
        // setRasterUniforms(*m_tiledCylinderRasterShader, camera);
        // m_tiledCylinderRasterShader->setInt("tilesX", m_config.tilesX);
        // m_tiledCylinderRasterShader->setInt("tilesY", m_config.tilesY);
        
        // glDispatchCompute(m_config.tilesX, m_config.tilesY, 1);
        // glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
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
// MAIN
// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "Hybrid GLSL Rasterization Pipeline" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    // =========================================================================
    // Configuration
    // =========================================================================
    
    HybridGLSLConfig config;
    config.computeDerivedValues();
    
    // =========================================================================
    // Window Setup
    // =========================================================================
    
    AppOpenGL window("Hybrid GLSL Pipeline", config.screenWidth, config.screenHeight, config.shown);
    
    // =========================================================================
    // Scene Loading
    // =========================================================================
    
    std::cout << "Loading scene..." << std::endl;
    
    std::vector<glm::vec4> spheres;
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
    // Camera Setup
    // =========================================================================
    
    Camera camera(config.screenWidth, config.screenHeight);
    camera.SetPosition(0.0f, 0.0f, 0.0f);
    
    CameraController cameraController(window, camera);
    
    // =========================================================================
    // Initialize Pipeline
    // =========================================================================
    
    HybridGLSLResources resources;
    resources.initialize(config, spheres, cylinders);
    
    HybridGLSLPipeline pipeline;
    pipeline.initialize(config);
    
    // =========================================================================
    // Canvas Setup
    // =========================================================================
    
    Canvas canvas(config.screenWidth, config.screenHeight);
    canvas.bindImage(0);
    
    // =========================================================================
    // Profiling Setup
    // =========================================================================
    
    Benchmark benchmark(cameraController, checkpoints);
    
    std::string basePath = "media/csv/hybrid_glsl/";
    std::string frameTimesPath = basePath + "frame_times.csv";
    std::string processTimesPath = basePath + "process_times.csv";
    
    Profiler profiler(window, frameTimesPath, processTimesPath,
                      config.sphereCount, config.cylinderCount, 0, config.timerDuration);
    
    int visibleSpheres = 0;
    int drawnSpheres = 0;
    int visibleCylinders = 0;
    int drawnCylinders = 0;
    
    // =========================================================================
    // GUI Setup
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
        
        pipeline.execute(camera, frustum, resources);
        
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
