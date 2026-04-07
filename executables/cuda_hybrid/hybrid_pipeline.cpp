/**
 * @file hybrid_pipeline.cpp
 * @brief Hybrid Rasterization Pipeline - CUDA Implementation
 * 
 * This executable implements a hybrid rasterization pipeline that combines:
 * - First version approach for small entities (1 thread per entity)
 * - Second version approach for large entities (tiled rendering)
 * 
 * Key design: Separate pipelines for spheres and cylinders
 * - Each type is processed completely before moving to the next
 * - Better cache locality and modularity
 * - Spheres processed first, then cylinders
 * 
 * Pipeline stages (per entity type):
 * 1. Frustum culling + BBox extraction + Size classification
 * 2. Radix sort of large billboards by depth (front-to-back)
 * 3. Tile binning for large entities
 * 4. Parallel execution of:
 *    a. Small entity rasterization (direct, 1 thread per entity)
 *    b. Large entity tiled rasterization (1 thread per pixel per tile)
 * 
 * Architecture follows SOLID principles with separate pipeline classes
 */

#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <chrono>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glad/glad.h>
#include <SDL3/SDL.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

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
#include "vis/gl/cu/storage_buffer_cu.h"
#include "vis/gl/texture.h"
#include "vis/gl/cu/canvas_cu.h"

// Include hybrid pipeline headers
#include "../../kernels/hybrid/hybrid_types.h"
#include "../../kernels/hybrid/sphere_pipeline.cuh"
#include "../../kernels/hybrid/cylinder_pipeline.cuh"

using uint = unsigned int;

// ============================================================================
// Configuration
// ============================================================================

struct HybridPipelineConfig {
    int screenWidth = 1024;
    int screenHeight = 1024;
    int sphereCount = 0;
    int cylinderCount = 0;
    bool shown = true;
    int smallEntityThreshold = SMALL_ENTITY_THRESHOLD;
    double timerDuration = 48.0;
    
    // Derived values
    int tilesX;
    int tilesY;
    
    void computeDerivedValues() {
        tilesX = (screenWidth + TILE_SIZE - 1) / TILE_SIZE;
        tilesY = (screenHeight + TILE_SIZE - 1) / TILE_SIZE;
    }
};

// ============================================================================
// External CUDA functions
// ============================================================================

extern "C" {
    // Screen clearing
    void launchScreenClear(
        cudaSurfaceObject_t outputImage,
        unsigned int* d_depthBuffer,
        unsigned int screenWidth,
        unsigned int screenHeight,
        float farPlane,
        cudaStream_t stream);
    
    // Upload constants
    void uploadHybridConstants(const HybridConstants& constants, cudaStream_t stream);
}

// ============================================================================
// Hybrid Pipeline Manager
// ============================================================================

class HybridPipelineManager {
public:
    HybridPipelineManager(const HybridPipelineConfig& config)
        : m_config(config)
    {
    }
    
    ~HybridPipelineManager() {
        cleanup();
    }
    
    void initialize(int sphereCount, int cylinderCount) {
        m_sphereCount = sphereCount;
        m_cylinderCount = cylinderCount;
        
        // Initialize sphere pipeline resources
        initSpherePipelineResources(
            &m_sphereResources, 
            sphereCount, 
            m_config.tilesX, 
            m_config.tilesY);
        
        // Initialize cylinder pipeline resources
        initCylinderPipelineResources(
            &m_cylinderResources, 
            cylinderCount, 
            m_config.tilesX, 
            m_config.tilesY);
        
        std::cout << "Pipeline initialized:" << std::endl;
        std::cout << "  Sphere pipeline: max " << sphereCount << " spheres" << std::endl;
        std::cout << "  Cylinder pipeline: max " << cylinderCount << " cylinders" << std::endl;
        std::cout << "  Tile grid: " << m_config.tilesX << "x" << m_config.tilesY << std::endl;
    }
    
    void cleanup() {
        freeSpherePipelineResources(&m_sphereResources);
        freeCylinderPipelineResources(&m_cylinderResources);
    }
    
    void executeFrame(
        const glm::vec4* d_spheres,
        const Cylinder* d_cylinders,
        unsigned int* d_depthBuffer,
        cudaSurfaceObject_t outputImage,
        Camera& camera,
        const Frustum& frustum,
        cudaStream_t stream)
    {
        nvtxRangePushA("Hybrid Pipeline Frame");
        
        // Upload constants for this frame
        uploadFrameConstants(camera, frustum, stream);
        
        // =====================================================================
        // PROCESS ALL SPHERES FIRST
        // =====================================================================
        resetSpherePipelineCounters(&m_sphereResources, stream);
        
        executeSpherePipeline(
            d_spheres,
            m_sphereCount,
            d_depthBuffer,
            outputImage,
            &m_sphereResources,
            stream);
        
        // =====================================================================
        // THEN PROCESS ALL CYLINDERS
        // =====================================================================
        resetCylinderPipelineCounters(&m_cylinderResources, stream);
        
        executeCylinderPipeline(
            d_cylinders,
            d_spheres,
            m_cylinderCount,
            d_depthBuffer,
            outputImage,
            &m_cylinderResources,
            stream);
        
        nvtxRangePop(); // Hybrid Pipeline Frame
    }
    
    // Statistics accessors
    void getSphereStats(unsigned int* frustumPassed, unsigned int* smallCount, unsigned int* largeCount) {
        getSpherePipelineStats(&m_sphereResources, frustumPassed, smallCount, largeCount);
    }
    
    void getCylinderStats(unsigned int* frustumPassed, unsigned int* smallCount, unsigned int* largeCount) {
        getCylinderPipelineStats(&m_cylinderResources, frustumPassed, smallCount, largeCount);
    }
    
private:
    void uploadFrameConstants(Camera& camera, const Frustum& frustum, cudaStream_t stream) {
        HybridConstants constants;
        
        constants.view = camera.getView();
        constants.proj = camera.getProjection();
        
        constants.frustumPlanes[0] = glm::vec4(frustum.topFace.normal, frustum.topFace.distance);
        constants.frustumPlanes[1] = glm::vec4(frustum.bottomFace.normal, frustum.bottomFace.distance);
        constants.frustumPlanes[2] = glm::vec4(frustum.rightFace.normal, frustum.rightFace.distance);
        constants.frustumPlanes[3] = glm::vec4(frustum.leftFace.normal, frustum.leftFace.distance);
        constants.frustumPlanes[4] = glm::vec4(frustum.farFace.normal, frustum.farFace.distance);
        constants.frustumPlanes[5] = glm::vec4(frustum.nearFace.normal, frustum.nearFace.distance);
        
        constants.front = camera.getFront();
        constants.up = camera.getUp();
        constants.right = camera.getRight();
        constants.cameraPos = camera.getPosition();
        
        constants.screenWidth = m_config.screenWidth;
        constants.screenHeight = m_config.screenHeight;
        constants.fov = camera.getFov();
        
        constants.sphereCount = m_sphereCount;
        constants.cylinderCount = m_cylinderCount;
        
        constants.tilesX = m_config.tilesX;
        constants.tilesY = m_config.tilesY;
        constants.totalTiles = m_config.tilesX * m_config.tilesY;
        
        constants.smallEntityThreshold = m_config.smallEntityThreshold;
        constants.benchmark = 1;
        
        uploadHybridConstants(constants, stream);
    }
    
    HybridPipelineConfig m_config;
    int m_sphereCount = 0;
    int m_cylinderCount = 0;
    
    SpherePipelineResources m_sphereResources;
    CylinderPipelineResources m_cylinderResources;
};

// ============================================================================
// Configuration Loader
// ============================================================================

HybridPipelineConfig loadConfiguration() {
    HybridPipelineConfig config;
    
    SceneSettings settings = SceneConfigLoader::loadDefault();
    SceneConfigLoader::applyToGlobals(settings);
    
    config.screenWidth = settings.screenWidth;
    config.screenHeight = settings.screenHeight;
    config.sphereCount = settings.sphereCount;
    config.cylinderCount = settings.cylinderCount;
    config.timerDuration = settings.timerDuration;
    config.smallEntityThreshold = SMALL_ENTITY_THRESHOLD;
    
    config.computeDerivedValues();
    
    return config;
}

// ============================================================================
// Main Application
// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << "=============================================" << std::endl;
    std::cout << "  Hybrid Rasterization Pipeline (CUDA)       " << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << std::endl;
    std::cout << "Design: Separate pipelines per entity type" << std::endl;
    std::cout << "  1. Process ALL spheres completely" << std::endl;
    std::cout << "  2. Process ALL cylinders completely" << std::endl;
    std::cout << std::endl;
    std::cout << "Each pipeline:" << std::endl;
    std::cout << "  - Small entities: Direct rasterization" << std::endl;
    std::cout << "  - Large entities: Sorted + Tiled rasterization" << std::endl;
    std::cout << std::endl;
    
    // Load configuration
    HybridPipelineConfig config = loadConfiguration();
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Screen: " << config.screenWidth << "x" << config.screenHeight << std::endl;
    std::cout << "  Tiles: " << config.tilesX << "x" << config.tilesY 
              << " (" << config.tilesX * config.tilesY << " total)" << std::endl;
    std::cout << "  Small entity threshold: " << config.smallEntityThreshold << " pixels^2" << std::endl;
    std::cout << std::endl;
    
    // Initialize CUDA
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "Error: No CUDA devices found!" << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "CUDA Device: " << prop.name << std::endl;
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  SM count: " << prop.multiProcessorCount << std::endl;
    std::cout << std::endl;
    
    cudaSetDevice(0);
    
    // Create window
    std::string title = "Hybrid Rasterization Pipeline (CUDA) - Separate Entity Pipelines";
    AppOpenGL window(title, config.screenWidth, config.screenHeight, config.shown);
    
    // Create camera
    Camera camera(config.screenWidth, config.screenHeight);
    camera.SetPosition(0.0f, 0.0f, 0.0f);
    
    CameraController camera_controller(window, camera);
    
    // Load scene
    std::vector<Sphere> spheres;
    std::vector<Cylinder> cylinders;
    getCompleteScene(spheres, config.sphereCount, cylinders, config.cylinderCount);
    const int actualSphereCount = static_cast<int>(spheres.size());
    const int actualCylinderCount = static_cast<int>(cylinders.size());
    
    std::cout << "Scene loaded:" << std::endl;
    std::cout << "  Spheres: " << actualSphereCount << std::endl;
    std::cout << "  Cylinders: " << actualCylinderCount << std::endl;
    std::cout << std::endl;
    
    // Get checkpoints for benchmark
    std::vector<std::pair<glm::vec3, glm::vec3>> checkpoints = 
        getCheckpoints(config.sphereCount, spheres, cylinders);
    
    // Create canvas with CUDA interop
    CanvasCUDA canvas(GL_TEXTURE_2D, GL_RGBA8, config.screenWidth, config.screenHeight);
    canvas.setFBO(GL_COLOR_ATTACHMENT0);
    canvas.setupDepthDataCUDA(config.screenWidth, config.screenHeight);
    
    // Get CUDA surface and depth buffer
    TextureCUDAWrapper& texWrapper = canvas.getCUDATextureWrapper();
    texWrapper.cudaMapResources();
    texWrapper.cudaCreateSurfaceObj();
    cudaSurfaceObject_t outputSurface = texWrapper.getSurfaceObject();
    unsigned int* depthBuffer = canvas.getDepthDataCUDA().getDepthBuffer();
    
    // Create GPU buffers for scene data
    StorageBuffer sphereBuffer(GL_SHADER_STORAGE_BUFFER, 
                               actualSphereCount * sizeof(Sphere), 6,
                               spheres.data(), GL_STATIC_DRAW);
    StorageBufferCUDAWrapper sphereBufferCUDA(sphereBuffer);
    sphereBufferCUDA.cudaMapResources();
    glm::vec4* d_spheres = sphereBufferCUDA.getDevicePointer<glm::vec4>();
    
    StorageBuffer cylinderBuffer(GL_SHADER_STORAGE_BUFFER, 
                                 actualCylinderCount * sizeof(Cylinder), 7,
                                 cylinders.data(), GL_STATIC_DRAW);
    StorageBufferCUDAWrapper cylinderBufferCUDA(cylinderBuffer);
    cylinderBufferCUDA.cudaMapResources();
    Cylinder* d_cylinders = cylinderBufferCUDA.getDevicePointer<Cylinder>();
    
    // Clear CPU vectors
    spheres.clear();
    cylinders.clear();
    
    // Initialize pipeline manager
    HybridPipelineManager pipelineManager(config);
    pipelineManager.initialize(actualSphereCount, actualCylinderCount);
    
    // Setup benchmark and profiler
    Benchmark benchmark(camera_controller, checkpoints);
    
    std::string basePath = "media/csv/cuda_hybrid/";
    std::string frameTimesPath = basePath + "frame_times.csv";
    std::string processTimesPath = basePath + "process_times.csv";
    
    // Statistics variables
    int visibleSpheres = actualSphereCount;
    int drawnSpheres = actualSphereCount;
    int visibleCylinders = actualCylinderCount;
    int drawnCylinders = actualCylinderCount;
    
    unsigned int sphereFrustum = 0, sphereSmall = 0, sphereLarge = 0;
    unsigned int cylFrustum = 0, cylSmall = 0, cylLarge = 0;
    
    Profiler profiler(window, frameTimesPath, processTimesPath,
                      actualSphereCount, actualCylinderCount, 0, config.timerDuration);
    
    // Setup GUI
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
    
    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Main loop
    std::cout << "Starting render loop..." << std::endl;
    std::cout << "  Press F10 for screenshot" << std::endl;
    std::cout << "  Press T to start profiling" << std::endl;
    std::cout << std::endl;
    
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    bool isRunning = true;
    
    while (isRunning) {
        nvtxRangePushA("Frame");
        
        // Update camera and benchmark
        camera_controller.cameraUpdate();
        benchmark.update();
        profiler.updateProfiler(benchmark.getCheckpointID(), 
                                visibleSpheres, drawnSpheres,
                                visibleCylinders, drawnCylinders);
        
        if (window.getInput().isKeyDown(Key::T)) {
            profiler.startSavingNextFrames(benchmark.getCheckpointID());
        }
        
        // Clear screen
        nvtxRangePushA("Screen Clear");
        launchScreenClear(outputSurface, depthBuffer, 
                          config.screenWidth, config.screenHeight,
                          camera.getFar(), stream);
        nvtxRangePop();
        
        // Create frustum for culling
        Frustum frustum(camera);
        
        // Execute hybrid pipeline (spheres first, then cylinders)
        pipelineManager.executeFrame(
            d_spheres, d_cylinders, 
            depthBuffer, outputSurface,
            camera, frustum, stream);
        
        // Wait for GPU
        cudaStreamSynchronize(stream);
        
        // Update statistics
        pipelineManager.getSphereStats(&sphereFrustum, &sphereSmall, &sphereLarge);
        pipelineManager.getCylinderStats(&cylFrustum, &cylSmall, &cylLarge);
        
        visibleSpheres = static_cast<int>(sphereFrustum);
        drawnSpheres = static_cast<int>(sphereSmall + sphereLarge);
        visibleCylinders = static_cast<int>(cylFrustum);
        drawnCylinders = static_cast<int>(cylSmall + cylLarge);
        
        // Display
        glBindFramebuffer(GL_READ_FRAMEBUFFER, canvas.getFramebuffer().getId());
        isRunning = window.update();
        glFinish();
        
        // Screenshot
        if (window.getInput().isKeyDown(Key::F10)) {
            std::string sshotName = "media/img/cuda_hybrid/frame_" + 
                                    std::to_string(benchmark.getCheckpointID()) + ".bmp";
            canvas.takeScreenshot(sshotName);
            std::cout << "Screenshot saved: " << sshotName << std::endl;
        }
        
        nvtxRangePop(); // Frame
    }
    
    // Cleanup
    std::cout << std::endl;
    std::cout << "Cleaning up..." << std::endl;
    
    cudaStreamDestroy(stream);
    
    texWrapper.cudaDestroySurfaceObj();
    texWrapper.cudaUnmapResources();
    sphereBufferCUDA.cudaUnmapResources();
    cylinderBufferCUDA.cudaUnmapResources();
    
    std::cout << "Done." << std::endl;
    
    return 0;
}
