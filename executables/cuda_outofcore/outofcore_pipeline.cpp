/**
 * @file outofcore_pipeline.cpp
 * @brief Out-of-core octree streaming pipeline executable.
 *
 * Implements the full render loop:
 *   1. Preprocess: Morton sort + block file + octree (once).
 *   2. Per frame:
 *      a) GPU: octree frustum culling (BFS by levels).
 *      b) GPU: depth+area computation, thrust sort, probabilistic occlusion.
 *      c) GPU: request generation, uniqueness via thrust sort+unique.
 *      d) CPU: LRU eviction + async block upload (double buffered).
 *      e) GPU: build active atom list, rasterise.
 *
 * References:
 *   - Sharma et al., "Scalable and portable visualization of large
 *     atomistic datasets", CPC 2004. (Hierarchical view frustum culling
 *     and probabilistic occlusion culling.)
 *   - Crassin et al., "GigaVoxels: Ray-Guided Streaming for Efficient
 *     and Detailed Voxel Rendering", I3D 2009. (LRU brick pool,
 *     CPU-GPU coordination.)
 */

#include <iostream>
#include <vector>
#include <algorithm>

#include <glm/glm.hpp>
#include <glad/glad.h>
#include <SDL3/SDL.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>

#include "appSDLGL.h"
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

#include "outofcore/outofcore_types.h"
#include "outofcore/preprocessor.h"
#include "outofcore/streaming_manager.h"

// ─────────────────── Extern kernel launchers ───────────────────

extern "C" void uploadOocConstants(const OocConstants& constants, cudaStream_t stream);
extern "C" void uploadOocRasterConstants(const OocConstants& constants, cudaStream_t stream);

extern "C" void launchScreenClearOoc(
    cudaSurfaceObject_t outputImage, unsigned int* d_depthBuffer,
    unsigned int screenWidth, unsigned int screenHeight,
    float farPlane, cudaStream_t stream);

extern "C" void launchOctreeFrustumCullLevel(
    const OocOctreeNode* d_octree, const unsigned int* d_inQueue,
    unsigned int inCount, unsigned int* d_outQueue, unsigned int* d_outCount,
    unsigned int* d_visibleBlockIds, unsigned int* d_visibleBlockCount,
    int maxVisibleBlocks, cudaStream_t stream);

extern "C" void launchComputeBlockDepthArea(
    const OocBlockMetadata* d_blockMeta, const unsigned int* d_visibleBlockIds,
    unsigned int visibleCount, OocBlockDepthInfo* d_depthInfo,
    cudaStream_t stream);

extern "C" void launchProbabilisticOcclusion(
    const OocBlockDepthInfo* d_sorted, unsigned int count,
    unsigned int* d_filteredBlockIds, unsigned int* d_filteredCount,
    cudaStream_t stream);

extern "C" void launchComputeBlockRequests(
    const unsigned int* d_filteredBlockIds, unsigned int filteredCount,
    const int32_t* d_blockSlotMap, unsigned int* d_requestBuffer,
    unsigned int* d_requestCount, int maxRequests, cudaStream_t stream);

extern "C" void launchBuildActiveAtomList(
    const glm::vec4* d_atomPool, const int32_t* d_blockSlotMap,
    const unsigned int* d_filteredBlockIds, unsigned int filteredCount,
    const unsigned int* d_blockAtomCounts, glm::vec4* d_activeAtoms,
    unsigned int* d_activeCount, cudaStream_t stream);

extern "C" void launchSphereRasterOoc(
    const glm::vec4* d_activeAtoms, unsigned int activeCount,
    unsigned int* d_depthBuffer, cudaSurfaceObject_t outputImage,
    cudaStream_t stream);

// ─────────────────── Thrust comparator ───────────────────

struct DepthInfoLess {
    __host__ __device__
    bool operator()(const OocBlockDepthInfo& a, const OocBlockDepthInfo& b) const {
        return a.depth < b.depth;
    }
};

// ─────────────────── Pipeline GPU resources ───────────────────

struct OocGpuResources {
    OocOctreeNode*    d_octree          = nullptr;
    OocBlockMetadata* d_blockMeta       = nullptr;
    unsigned int*     d_blockAtomCounts = nullptr;

    // BFS queues
    unsigned int* d_queueA       = nullptr;
    unsigned int* d_queueB       = nullptr;
    unsigned int* d_queueCountA  = nullptr;
    unsigned int* d_queueCountB  = nullptr;

    unsigned int* d_visibleBlockIds   = nullptr;
    unsigned int* d_visibleBlockCount = nullptr;

    OocBlockDepthInfo* d_depthInfo    = nullptr;

    unsigned int* d_filteredBlockIds  = nullptr;
    unsigned int* d_filteredCount     = nullptr;

    unsigned int* d_requestBuffer     = nullptr;
    unsigned int* d_requestCount      = nullptr;

    glm::vec4*    d_activeAtoms       = nullptr;
    unsigned int* d_activeCount       = nullptr;

    int maxBlocks     = 0;
    int maxOctreeNodes = 0;
    int maxActiveAtoms = 0;
};

static void allocGpuResources(OocGpuResources& r,
                               const ooc::PreprocessResult& prep,
                               int maxActiveAtoms)
{
    int numBlocks = static_cast<int>(prep.blocks.size());
    int numNodes  = static_cast<int>(prep.octreeNodes.size());
    r.maxBlocks      = numBlocks;
    r.maxOctreeNodes = numNodes;
    r.maxActiveAtoms = maxActiveAtoms;

    cudaMalloc(&r.d_octree,    numNodes  * sizeof(OocOctreeNode));
    cudaMalloc(&r.d_blockMeta, numBlocks * sizeof(OocBlockMetadata));
    cudaMalloc(&r.d_blockAtomCounts, numBlocks * sizeof(unsigned int));

    cudaMemcpy(r.d_octree, prep.octreeNodes.data(),
               numNodes * sizeof(OocOctreeNode), cudaMemcpyHostToDevice);
    cudaMemcpy(r.d_blockMeta, prep.blocks.data(),
               numBlocks * sizeof(OocBlockMetadata), cudaMemcpyHostToDevice);

    std::vector<unsigned int> counts(numBlocks);
    for (int i = 0; i < numBlocks; i++) counts[i] = prep.blocks[i].atomCount;
    cudaMemcpy(r.d_blockAtomCounts, counts.data(),
               numBlocks * sizeof(unsigned int), cudaMemcpyHostToDevice);

    int maxQueue = numNodes + 8;
    cudaMalloc(&r.d_queueA,      maxQueue  * sizeof(unsigned int));
    cudaMalloc(&r.d_queueB,      maxQueue  * sizeof(unsigned int));
    cudaMalloc(&r.d_queueCountA, sizeof(unsigned int));
    cudaMalloc(&r.d_queueCountB, sizeof(unsigned int));

    cudaMalloc(&r.d_visibleBlockIds,   numBlocks * sizeof(unsigned int));
    cudaMalloc(&r.d_visibleBlockCount, sizeof(unsigned int));

    cudaMalloc(&r.d_depthInfo,        numBlocks * sizeof(OocBlockDepthInfo));
    cudaMalloc(&r.d_filteredBlockIds, numBlocks * sizeof(unsigned int));
    cudaMalloc(&r.d_filteredCount,    sizeof(unsigned int));

    cudaMalloc(&r.d_requestBuffer, OOC_MAX_REQUESTS_PER_FRAME * sizeof(unsigned int));
    cudaMalloc(&r.d_requestCount,  sizeof(unsigned int));

    cudaMalloc(&r.d_activeAtoms, maxActiveAtoms * sizeof(glm::vec4));
    cudaMalloc(&r.d_activeCount, sizeof(unsigned int));
}

static void freeGpuResources(OocGpuResources& r) {
    cudaFree(r.d_octree);
    cudaFree(r.d_blockMeta);
    cudaFree(r.d_blockAtomCounts);
    cudaFree(r.d_queueA);
    cudaFree(r.d_queueB);
    cudaFree(r.d_queueCountA);
    cudaFree(r.d_queueCountB);
    cudaFree(r.d_visibleBlockIds);
    cudaFree(r.d_visibleBlockCount);
    cudaFree(r.d_depthInfo);
    cudaFree(r.d_filteredBlockIds);
    cudaFree(r.d_filteredCount);
    cudaFree(r.d_requestBuffer);
    cudaFree(r.d_requestCount);
    cudaFree(r.d_activeAtoms);
    cudaFree(r.d_activeCount);
}

// ═════════════════════════════════════════════════════════
// main
// ═════════════════════════════════════════════════════════

int main(int argc, char* argv[]) {
    std::cout << "=============================================" << std::endl;
    std::cout << "  Out-of-Core Octree Streaming Pipeline      " << std::endl;
    std::cout << "=============================================" << std::endl;

    // ── Load configuration ──
    SceneSettings settings = SceneConfigLoader::loadDefault();
    SceneConfigLoader::applyToGlobals(settings);
    int screenWidth  = settings.screenWidth;
    int screenHeight = settings.screenHeight;
    int sphereCount  = settings.sphereCount;

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return 1;
    }
    cudaSetDevice(0);

    // ── Load scene data on CPU ──
    std::vector<glm::vec4> spheres;
    std::vector<CylinderIndex> cylinders;
    getCompleteScene(spheres, sphereCount, cylinders, 0);
    sphereCount = static_cast<int>(spheres.size());
    std::cout << "Loaded " << sphereCount << " atoms." << std::endl;

    // ── Preprocess (once) ──
    ooc::PreprocessResult prep = ooc::preprocess(
        spheres.data(), static_cast<uint32_t>(sphereCount), "ooc_data");

    int totalBlocks = static_cast<int>(prep.blocks.size());
    int poolSlots   = std::min(OOC_MAX_BLOCK_POOL_SLOTS, totalBlocks);
    int maxActiveAtoms = poolSlots * OOC_ATOMS_PER_BLOCK;

    // ── Window + Camera ──
    AppOpenGL window("Out-of-Core Octree Streaming",
                     screenWidth, screenHeight, true);
    Camera camera(screenWidth, screenHeight);
    camera.SetPosition(0.0f, 0.0f, 0.0f);
    CameraController cameraController(window, camera);

    std::vector<std::pair<glm::vec3, glm::vec3>> checkpoints =
        getCheckpoints(sphereCount, spheres, cylinders);

    spheres.clear();
    cylinders.clear();

    // ── Canvas (GL-CUDA interop) ──
    CanvasCUDA canvas(GL_TEXTURE_2D, GL_RGBA8, screenWidth, screenHeight);
    canvas.setFBO(GL_COLOR_ATTACHMENT0);
    canvas.setupDepthDataCUDA(screenWidth, screenHeight);

    TextureCUDAWrapper& texWrapper = canvas.getCUDATextureWrapper();
    texWrapper.cudaMapResources();
    texWrapper.cudaCreateSurfaceObj();
    cudaSurfaceObject_t outputSurface = texWrapper.getSurfaceObject();
    unsigned int* depthBuffer = canvas.getDepthDataCUDA().getDepthBuffer();

    // ── GPU resources ──
    OocGpuResources gpu;
    allocGpuResources(gpu, prep, maxActiveAtoms);

    // ── Streaming manager ──
    ooc::StreamingManager streamMgr;
    streamMgr.initialize(poolSlots, totalBlocks,
                         prep.blockFilePath, prep.blocks);

    // ── Benchmark / profiler ──
    Benchmark benchmark(cameraController, checkpoints);
    int visibleAtoms = 0, drawnAtoms = 0;
    int dummyCyls = 0;
    Profiler profiler(window, "media/csv/cuda_outofcore/frame_times.csv",
                      "media/csv/cuda_outofcore/process_times.csv",
                      sphereCount, 0, 0, settings.timerDuration);

    std::unordered_map<std::string, int*> sceneData = {
        {"Screen width",   &screenWidth},
        {"Screen height",  &screenHeight},
        {"Total atoms",    &sphereCount},
        {"Visible atoms",  &visibleAtoms},
        {"Drawn atoms",    &drawnAtoms},
        {"Total blocks",   &totalBlocks},
        {"Pool slots",     &poolSlots}
    };
    window.setupSceneInfoGui("Scene Info", sceneData);
    window.setupCameraGui("Camera Info", &camera);
    window.setupInputInfoGui("Input Info");
    window.setupBenchmarkInfoGui("Benchmark", &benchmark);

    // ── CUDA streams ──
    cudaStream_t renderStream, uploadStream;
    cudaStreamCreate(&renderStream);
    cudaStreamCreate(&uploadStream);

    // Pinned host buffers for readback
    unsigned int* h_visibleBlockCount;
    unsigned int* h_filteredCount;
    unsigned int* h_requestCount;
    unsigned int* h_activeCount;
    unsigned int* h_requestBuffer;
    unsigned int* h_filteredBlockIds;
    unsigned int* h_queueCount;

    cudaHostAlloc(&h_visibleBlockCount, sizeof(unsigned int), cudaHostAllocDefault);
    cudaHostAlloc(&h_filteredCount,     sizeof(unsigned int), cudaHostAllocDefault);
    cudaHostAlloc(&h_requestCount,      sizeof(unsigned int), cudaHostAllocDefault);
    cudaHostAlloc(&h_activeCount,       sizeof(unsigned int), cudaHostAllocDefault);
    cudaHostAlloc(&h_queueCount,        sizeof(unsigned int), cudaHostAllocDefault);
    cudaHostAlloc(&h_requestBuffer,
                  OOC_MAX_REQUESTS_PER_FRAME * sizeof(unsigned int),
                  cudaHostAllocDefault);
    cudaHostAlloc(&h_filteredBlockIds,
                  totalBlocks * sizeof(unsigned int),
                  cudaHostAllocDefault);

    uint64_t frameId = 0;

    // ── Render loop ──
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    bool isRunning = true;

    while (isRunning) {
        cameraController.cameraUpdate();
        benchmark.update();
        profiler.updateProfiler(benchmark.getCheckpointID(),
                                visibleAtoms, drawnAtoms, 0, 0);

        Frustum frustum(camera);

        // Fill OocConstants
        OocConstants cst{};
        cst.view     = camera.getView();
        cst.proj     = camera.getProjection();
        cst.viewProj = cst.proj * cst.view;
        cst.frustumPlanes[0] = glm::vec4(frustum.topFace.normal,    frustum.topFace.distance);
        cst.frustumPlanes[1] = glm::vec4(frustum.bottomFace.normal, frustum.bottomFace.distance);
        cst.frustumPlanes[2] = glm::vec4(frustum.rightFace.normal,  frustum.rightFace.distance);
        cst.frustumPlanes[3] = glm::vec4(frustum.leftFace.normal,   frustum.leftFace.distance);
        cst.frustumPlanes[4] = glm::vec4(frustum.farFace.normal,    frustum.farFace.distance);
        cst.frustumPlanes[5] = glm::vec4(frustum.nearFace.normal,   frustum.nearFace.distance);
        cst.cameraPos     = camera.getPosition();
        cst.screenWidth   = screenWidth;
        cst.screenHeight  = screenHeight;
        cst.fov           = camera.getFov();
        cst.totalBlocks   = totalBlocks;
        cst.octreeNodeCount = gpu.maxOctreeNodes;
        cst.visibilityThreshold = OOC_VISIBILITY_THRESHOLD;
        cst.maxPoolSlots  = poolSlots;

        uploadOocConstants(cst, renderStream);
        uploadOocRasterConstants(cst, renderStream);

        // ── Clear screen ──
        launchScreenClearOoc(outputSurface, depthBuffer,
                             screenWidth, screenHeight,
                             camera.getFar(), renderStream);

        // ── Phase 1: Octree BFS frustum culling ──
        cudaMemsetAsync(gpu.d_visibleBlockCount, 0, sizeof(unsigned int), renderStream);

        // Seed queue with root node (index 0)
        unsigned int rootIdx = 0;
        cudaMemcpyAsync(gpu.d_queueA, &rootIdx, sizeof(unsigned int),
                        cudaMemcpyHostToDevice, renderStream);
        unsigned int one = 1;
        cudaMemcpyAsync(gpu.d_queueCountA, &one, sizeof(unsigned int),
                        cudaMemcpyHostToDevice, renderStream);

        unsigned int* inQueue  = gpu.d_queueA;
        unsigned int* outQueue = gpu.d_queueB;
        unsigned int* inCount  = gpu.d_queueCountA;
        unsigned int* outCount = gpu.d_queueCountB;

        for (int level = 0; level < OOC_MAX_OCTREE_DEPTH + 1; level++) {
            cudaMemsetAsync(outCount, 0, sizeof(unsigned int), renderStream);

            cudaMemcpyAsync(h_queueCount, inCount, sizeof(unsigned int),
                            cudaMemcpyDeviceToHost, renderStream);
            cudaStreamSynchronize(renderStream);
            unsigned int currentCount = *h_queueCount;
            if (currentCount == 0) break;

            launchOctreeFrustumCullLevel(
                gpu.d_octree, inQueue, currentCount, outQueue, outCount,
                gpu.d_visibleBlockIds, gpu.d_visibleBlockCount,
                gpu.maxBlocks, renderStream);

            std::swap(inQueue, outQueue);
            std::swap(inCount, outCount);
        }

        cudaMemcpyAsync(h_visibleBlockCount, gpu.d_visibleBlockCount,
                        sizeof(unsigned int), cudaMemcpyDeviceToHost, renderStream);
        cudaStreamSynchronize(renderStream);
        unsigned int numVisible = *h_visibleBlockCount;
        visibleAtoms = static_cast<int>(numVisible) * OOC_ATOMS_PER_BLOCK;

        if (numVisible > 0) {
            // ── Phase 2: Depth + area, sort, probabilistic occlusion ──
            launchComputeBlockDepthArea(
                gpu.d_blockMeta, gpu.d_visibleBlockIds, numVisible,
                gpu.d_depthInfo, renderStream);

            thrust::device_ptr<OocBlockDepthInfo> depthPtr(gpu.d_depthInfo);
            thrust::sort(thrust::cuda::par.on(renderStream),
                         depthPtr, depthPtr + numVisible, DepthInfoLess());

            cudaMemsetAsync(gpu.d_filteredCount, 0, sizeof(unsigned int), renderStream);
            launchProbabilisticOcclusion(
                gpu.d_depthInfo, numVisible,
                gpu.d_filteredBlockIds, gpu.d_filteredCount, renderStream);

            cudaMemcpyAsync(h_filteredCount, gpu.d_filteredCount,
                            sizeof(unsigned int), cudaMemcpyDeviceToHost, renderStream);
            cudaStreamSynchronize(renderStream);
            unsigned int numFiltered = *h_filteredCount;

            if (numFiltered > 0) {
                // ── Phase 3: Request generation ──
                cudaMemsetAsync(gpu.d_requestCount, 0, sizeof(unsigned int), renderStream);
                launchComputeBlockRequests(
                    gpu.d_filteredBlockIds, numFiltered,
                    streamMgr.getReadSlotMap(),
                    gpu.d_requestBuffer, gpu.d_requestCount,
                    OOC_MAX_REQUESTS_PER_FRAME, renderStream);

                // Sort + unique on request buffer
                cudaMemcpyAsync(h_requestCount, gpu.d_requestCount,
                                sizeof(unsigned int), cudaMemcpyDeviceToHost, renderStream);
                cudaStreamSynchronize(renderStream);
                unsigned int numRequests = std::min(*h_requestCount,
                    static_cast<unsigned int>(OOC_MAX_REQUESTS_PER_FRAME));

                if (numRequests > 0) {
                    thrust::device_ptr<unsigned int> reqPtr(gpu.d_requestBuffer);
                    thrust::sort(thrust::cuda::par.on(renderStream),
                                 reqPtr, reqPtr + numRequests);
                    auto newEnd = thrust::unique(thrust::cuda::par.on(renderStream),
                                                reqPtr, reqPtr + numRequests);
                    numRequests = static_cast<unsigned int>(newEnd - reqPtr);

                    cudaMemcpyAsync(h_requestBuffer, gpu.d_requestBuffer,
                                    numRequests * sizeof(unsigned int),
                                    cudaMemcpyDeviceToHost, renderStream);
                    cudaStreamSynchronize(renderStream);
                }

                // Read back filtered block IDs for LRU refresh
                cudaMemcpyAsync(h_filteredBlockIds, gpu.d_filteredBlockIds,
                                numFiltered * sizeof(unsigned int),
                                cudaMemcpyDeviceToHost, renderStream);
                cudaStreamSynchronize(renderStream);

                // ── Phase 4: CPU streaming (upload to write buffer) ──
                streamMgr.processRequests(
                    h_requestBuffer, numRequests,
                    h_filteredBlockIds, numFiltered,
                    frameId, uploadStream);
                cudaStreamSynchronize(uploadStream);

                // ── Phase 5: Build active atom list + render ──
                cudaMemsetAsync(gpu.d_activeCount, 0, sizeof(unsigned int), renderStream);
                launchBuildActiveAtomList(
                    streamMgr.getReadAtomPool(),
                    streamMgr.getReadSlotMap(),
                    gpu.d_filteredBlockIds, numFiltered,
                    gpu.d_blockAtomCounts,
                    gpu.d_activeAtoms, gpu.d_activeCount, renderStream);

                cudaMemcpyAsync(h_activeCount, gpu.d_activeCount,
                                sizeof(unsigned int), cudaMemcpyDeviceToHost, renderStream);
                cudaStreamSynchronize(renderStream);
                unsigned int activeCount = *h_activeCount;
                drawnAtoms = static_cast<int>(activeCount);

                if (activeCount > 0) {
                    launchSphereRasterOoc(
                        gpu.d_activeAtoms, activeCount,
                        depthBuffer, outputSurface, renderStream);
                    cudaStreamSynchronize(renderStream);
                }
            }
        }

        streamMgr.swapBuffers();

        glBindFramebuffer(GL_READ_FRAMEBUFFER, canvas.getFramebuffer().getId());
        isRunning = window.update();
        frameId++;
    }

    // ── Cleanup ──
    cudaStreamDestroy(renderStream);
    cudaStreamDestroy(uploadStream);
    texWrapper.cudaUnmapResources();
    texWrapper.cudaDestroySurfaceObj();
    freeGpuResources(gpu);
    streamMgr.destroy();

    cudaFreeHost(h_visibleBlockCount);
    cudaFreeHost(h_filteredCount);
    cudaFreeHost(h_requestCount);
    cudaFreeHost(h_activeCount);
    cudaFreeHost(h_requestBuffer);
    cudaFreeHost(h_filteredBlockIds);
    cudaFreeHost(h_queueCount);

    std::cout << "Done." << std::endl;
    return 0;
}
