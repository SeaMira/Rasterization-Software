/**
 * @file hybrid_binning_pipeline.cpp
 * @brief Hybrid CUDA pipeline with binning-by-sort (no linked lists).
 * Uses CUB DeviceRadixSort + DeviceRunLengthEncode for tile binning.
 */

#include <iostream>
#include <memory>
#include <vector>
#include <unordered_map>

#include <glm/glm.hpp>
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
#include "ux/camera.h"
#include "ux/camera_controller.h"
#include "algorithms/frustum_cull.h"
#include "ux/cinematic/benchmark.h"
#include "ux/profiler/profiler.h"
#include "vis/gl/frame_buffer.h"
#include "vis/gl/storage_buffer.h"
#include "vis/gl/cu/storage_buffer_cu.h"
#include "vis/gl/texture.h"
#include "vis/gl/cu/canvas_cu.h"

#include "../../kernels/hybrid_binning/hybrid_binning_types.h"
#include "../../kernels/hybrid_binning/sphere_pipeline_binning.cuh"
#include "../../kernels/hybrid_binning/cylinder_pipeline_binning.cuh"

using uint = unsigned int;

struct HybridBinningConfig {
    int screenWidth = 1024;
    int screenHeight = 1024;
    int sphereCount = 0;
    int cylinderCount = 0;
    bool shown = true;
    int smallEntityThreshold = SMALL_ENTITY_THRESHOLD;
    /** Average entities per tile for pair buffer size: buffer = avgEntitiesPerTile * totalTiles */
    int avgEntitiesPerTile = AVG_ENTITIES_PER_TILE;
    double timerDuration = 48.0;
    int tilesX;
    int tilesY;
    int totalTiles;
    void computeDerivedValues() {
        tilesX = (screenWidth + TILE_SIZE - 1) / TILE_SIZE;
        tilesY = (screenHeight + TILE_SIZE - 1) / TILE_SIZE;
        totalTiles = tilesX * tilesY;
    }
};

extern "C" void launchScreenClear(
    cudaSurfaceObject_t outputImage,
    unsigned int* d_depthBuffer,
    unsigned int screenWidth,
    unsigned int screenHeight,
    float farPlane,
    cudaStream_t stream);
extern "C" void uploadHybridConstants(const HybridConstants& constants, cudaStream_t stream);

class HybridBinningPipelineManager {
public:
    explicit HybridBinningPipelineManager(const HybridBinningConfig& config) : m_config(config) {}
    ~HybridBinningPipelineManager() {
        freeSphereBinningResources(&m_sphereResources);
        freeCylinderBinningResources(&m_cylinderResources);
    }

    void initialize(int sphereCount, int cylinderCount) {
        m_sphereCount = sphereCount;
        m_cylinderCount = cylinderCount;
        initSphereBinningResources(&m_sphereResources, sphereCount, m_config.tilesX, m_config.tilesY, m_config.totalTiles, m_config.avgEntitiesPerTile);
        initCylinderBinningResources(&m_cylinderResources, cylinderCount, m_config.tilesX, m_config.tilesY, m_config.totalTiles, m_config.avgEntitiesPerTile);
        std::cout << "Binning pipeline initialized: " << sphereCount << " spheres, " << cylinderCount << " cylinders" << std::endl;
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
        nvtxRangePushA("Hybrid Binning Frame");
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
        constants.totalTiles = m_config.totalTiles;
        constants.smallEntityThreshold = m_config.smallEntityThreshold;
        constants.benchmark = 1;

        // Pre-compute screen ray casting vectors (camera space)
        {
            float aspectRatio = (float)m_config.screenWidth / (float)m_config.screenHeight;
            float fovRad = glm::radians(camera.getFov());
            float fovTan = tanf(fovRad * 0.5f);
            float halfFovTan = fovTan * aspectRatio;

            glm::vec3 camFront = camera.getFront();
            glm::vec3 camUp    = camera.getUp();
            glm::vec3 camRight = camera.getRight();

            // Corner ray directions in world space
            glm::vec3 corner00 = glm::normalize(-halfFovTan * camRight - fovTan * camUp + camFront);
            glm::vec3 corner10 = glm::normalize( halfFovTan * camRight - fovTan * camUp + camFront);
            glm::vec3 corner01 = glm::normalize(-halfFovTan * camRight + fovTan * camUp + camFront);

            // Transform to camera space via the 3x3 rotation part of the view matrix
            glm::mat3 viewRot = glm::mat3(camera.getView());
            glm::vec3 wCorner00 = viewRot * corner00;
            glm::vec3 wCorner10 = viewRot * corner10;
            glm::vec3 wCorner01 = viewRot * corner01;

            constants.rayStart = wCorner00;
            constants.dx = (wCorner10 - wCorner00) / (float)m_config.screenWidth;
            constants.dy = (wCorner01 - wCorner00) / (float)m_config.screenHeight;
            constants.pxScale = 2.0f * fovTan / (float)m_config.screenHeight;
        }

        nvtxRangePushA("Upload Constants");
        uploadHybridConstants(constants, stream);
        nvtxRangePop();

        nvtxRangePushA("Reset Counters");
        resetSphereBinningCounters(&m_sphereResources, stream);
        resetCylinderBinningCounters(&m_cylinderResources, stream);
        nvtxRangePop();

        nvtxRangePushA("Screen Clear");
        launchScreenClear(outputImage, d_depthBuffer, m_config.screenWidth, m_config.screenHeight, camera.getFar(), stream);
        nvtxRangePop();

        nvtxRangePushA("Sphere Pipeline Binning");
        executeSpherePipelineBinning(d_spheres, m_sphereCount, d_depthBuffer, outputImage, &m_sphereResources, stream);
        nvtxRangePop();

        nvtxRangePushA("Cylinder Pipeline Binning");
        executeCylinderPipelineBinning(d_cylinders, d_spheres, m_cylinderCount, d_depthBuffer, outputImage, &m_cylinderResources, stream);
        nvtxRangePop();

        nvtxRangePop();
    }

    void getSphereStats(unsigned int* frustumPassed, unsigned int* smallCount, unsigned int* largeCount) {
        getSphereBinningStats(&m_sphereResources, frustumPassed, smallCount, largeCount);
    }
    void getCylinderStats(unsigned int* frustumPassed, unsigned int* smallCount, unsigned int* largeCount) {
        getCylinderBinningStats(&m_cylinderResources, frustumPassed, smallCount, largeCount);
    }

    void setSmallEntityThreshold(int threshold) {
        m_config.smallEntityThreshold = threshold;
    }

private:
    HybridBinningConfig m_config;
    int m_sphereCount = 0;
    int m_cylinderCount = 0;
    SphereBinningResources m_sphereResources;
    CylinderBinningResources m_cylinderResources;
};

HybridBinningConfig loadConfiguration() {
    HybridBinningConfig config;
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

int main(int argc, char* argv[]) {
    std::cout << "=============================================" << std::endl;
    std::cout << "  Hybrid CUDA Binning Pipeline              " << std::endl;
    std::cout << "  (Sort-based tile binning, no linked lists)" << std::endl;
    std::cout << "=============================================" << std::endl;

    HybridBinningConfig config = loadConfiguration();
    std::cout << "Screen: " << config.screenWidth << "x" << config.screenHeight
              << "  Tiles: " << config.tilesX << "x" << config.tilesY << std::endl;

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return 1;
    }
    cudaSetDevice(0);

    AppOpenGL window("Hybrid CUDA Binning", config.screenWidth, config.screenHeight, config.shown);
    Camera camera(config.screenWidth, config.screenHeight);
    camera.SetPosition(0.0f, 0.0f, 0.0f);
    CameraController cameraController(window, camera);

    std::vector<Sphere> spheres;
    std::vector<Cylinder> cylinders;
    getCompleteScene(spheres, config.sphereCount, cylinders, config.cylinderCount);
    const int actualSphereCount = static_cast<int>(spheres.size());
    const int actualCylinderCount = static_cast<int>(cylinders.size());
    std::cout << "Scene: " << actualSphereCount << " spheres, " << actualCylinderCount << " cylinders" << std::endl;

    std::vector<std::pair<glm::vec3, glm::vec3>> checkpoints = getCheckpoints(config.sphereCount, spheres, cylinders);
    CanvasCUDA canvas(GL_TEXTURE_2D, GL_RGBA8, config.screenWidth, config.screenHeight);
    canvas.setFBO(GL_COLOR_ATTACHMENT0);
    canvas.setupDepthDataCUDA(config.screenWidth, config.screenHeight);

    TextureCUDAWrapper& texWrapper = canvas.getCUDATextureWrapper();
    texWrapper.cudaMapResources();
    texWrapper.cudaCreateSurfaceObj();
    cudaSurfaceObject_t outputSurface = texWrapper.getSurfaceObject();
    unsigned int* depthBuffer = canvas.getDepthDataCUDA().getDepthBuffer();

    StorageBuffer sphereBuffer(GL_SHADER_STORAGE_BUFFER, actualSphereCount * sizeof(Sphere), 6, spheres.data(), GL_STATIC_DRAW);
    StorageBufferCUDAWrapper sphereBufferCUDA(sphereBuffer);
    sphereBufferCUDA.cudaMapResources();
    glm::vec4* d_spheres = sphereBufferCUDA.getDevicePointer<glm::vec4>();

    StorageBuffer cylinderBuffer(GL_SHADER_STORAGE_BUFFER, actualCylinderCount * sizeof(Cylinder), 7, cylinders.data(), GL_STATIC_DRAW);
    StorageBufferCUDAWrapper cylinderBufferCUDA(cylinderBuffer);
    cylinderBufferCUDA.cudaMapResources();
    Cylinder* d_cylinders = cylinderBufferCUDA.getDevicePointer<Cylinder>();

    spheres.clear();
    cylinders.clear();

    HybridBinningPipelineManager pipeline(config);
    pipeline.initialize(actualSphereCount, actualCylinderCount);

    Benchmark benchmark(cameraController, checkpoints);
    int visibleSpheres = actualSphereCount, drawnSpheres = actualSphereCount;
    int visibleCylinders = actualCylinderCount, drawnCylinders = actualCylinderCount;
    unsigned int sphereFrustum = 0, sphereSmall = 0, sphereLarge = 0;
    unsigned int cylFrustum = 0, cylSmall = 0, cylLarge = 0;
    Profiler profiler(window, "media/csv/cuda_hybrid_binning/frame_times.csv",
                      "media/csv/cuda_hybrid_binning/process_times.csv",
                      actualSphereCount, actualCylinderCount, 0, config.timerDuration);

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

    int tileSize = TILE_SIZE;
    int maxEntitiesPerTile = config.avgEntitiesPerTile;
    window.setupPipelineConfigGui("Pipeline Config",
        config.smallEntityThreshold, tileSize,
        maxEntitiesPerTile, config.tilesX, config.tilesY);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    bool isRunning = true;
    while (isRunning) {
        cameraController.cameraUpdate();
        benchmark.update();
        profiler.updateProfiler(benchmark.getCheckpointID(), visibleSpheres, drawnSpheres, visibleCylinders, drawnCylinders);

        Frustum frustum(camera);
        pipeline.setSmallEntityThreshold(config.smallEntityThreshold);
        pipeline.executeFrame(d_spheres, d_cylinders, depthBuffer, outputSurface, camera, frustum, stream);
        cudaStreamSynchronize(stream);  // ensure all kernels and D2H copies finish before reading stats

        pipeline.getSphereStats(&sphereFrustum, &sphereSmall, &sphereLarge);
        pipeline.getCylinderStats(&cylFrustum, &cylSmall, &cylLarge);
        visibleSpheres = static_cast<int>(sphereFrustum);
        drawnSpheres = static_cast<int>(sphereSmall + sphereLarge);
        visibleCylinders = static_cast<int>(cylFrustum);
        drawnCylinders = static_cast<int>(cylSmall + cylLarge);

        glBindFramebuffer(GL_READ_FRAMEBUFFER, canvas.getFramebuffer().getId());
        isRunning = window.update();
        glFinish();
    }

    cudaStreamDestroy(stream);
    texWrapper.cudaUnmapResources();
    texWrapper.cudaDestroySurfaceObj();
    sphereBufferCUDA.cudaUnmapResources();
    cylinderBufferCUDA.cudaUnmapResources();
    std::cout << "Done." << std::endl;
    return 0;
}
