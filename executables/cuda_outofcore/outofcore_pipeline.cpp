/**
 * @file outofcore_pipeline.cpp
 * @brief Out-of-core octree streaming pipeline executable.
 *
 * Implements the full render loop:
 *   1. Preprocess: Morton sort + block file + octree (once).
 *   2. Per frame:
 *      a) GPU: octree frustum culling (BFS by levels).
 *      b) GPU: depth+area computation, thrust sort, occlusion culling.
 *      c) GPU: request generation, uniqueness via thrust sort+unique.
 *      d) CPU: LRU eviction + async block upload (double buffered).
 *      e) GPU: build active atom list, rasterise.
 *
 * Occlusion culling method is selectable via scene_config.json:
 *   "none"                  — no occlusion culling
 *   "probabilistic"         — Atomsviewer probabilistic (real density)
 *   "probabilistic_overlap" — same but with screen-space overlap check
 *   "hiz"                   — HiZ per-block occlusion (previous frame)
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <limits>
#include <cstdint>

#include <glm/glm.hpp>
#include <glad/glad.h>
#include <SDL3/SDL.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
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
#include "vis/gl/cu/depth_downsample_cu.h"

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
    const OocOctreeNode* d_octree, const unsigned int* d_blockIndexBuffer,
    const unsigned int* d_inQueue, unsigned int inCount,
    unsigned int* d_outQueue, unsigned int* d_outCount,
    unsigned int* d_visibleBlockIds, unsigned int* d_visibleBlockCount,
    int maxVisibleBlocks, cudaStream_t stream);

extern "C" void launchOctreeFrustumCullLevelIndirect(
    const OocOctreeNode* d_octree, const unsigned int* d_blockIndexBuffer,
    const unsigned int* d_inQueue, const unsigned int* d_inCount,
    unsigned int* d_outQueue, unsigned int* d_outCount,
    unsigned int* d_visibleBlockIds, unsigned int* d_visibleBlockCount,
    int maxVisibleBlocks, int maxQueueSize, cudaStream_t stream);

extern "C" void launchOctreeFrustumCullLevelLod(
    const OocOctreeNode* d_octree, const unsigned int* d_blockIndexBuffer,
    const unsigned int* d_inQueue, const unsigned int* d_inCount,
    unsigned int* d_outQueue, unsigned int* d_outCount,
    unsigned int* d_visibleBlockIds, unsigned int* d_visibleBlockCount,
    int maxVisibleBlocks, int maxQueueSize,
    const glm::vec4* d_lodAtomBuffer, glm::vec4* d_lodActiveAtoms,
    unsigned int* d_lodActiveCount, int maxLodAtoms,
    cudaStream_t stream);

extern "C" void launchComputeBlockDepthArea(
    const OocBlockMetadata* d_blockMeta, const unsigned int* d_visibleBlockIds,
    unsigned int visibleCount, OocBlockDepthInfo* d_depthInfo,
    cudaStream_t stream);

extern "C" void launchNoOcclusionPassthrough(
    const OocBlockDepthInfo* d_depthInfo, unsigned int count,
    unsigned int* d_filteredBlockIds, unsigned int* d_filteredCount,
    cudaStream_t stream);

extern "C" void launchProbabilisticOcclusion(
    const OocBlockDepthInfo* d_sorted, unsigned int count,
    unsigned int* d_filteredBlockIds, unsigned int* d_filteredCount,
    cudaStream_t stream);

extern "C" void launchProbabilisticOcclusionOverlap(
    const OocBlockDepthInfo* d_sorted, unsigned int count,
    unsigned int* d_filteredBlockIds, unsigned int* d_filteredCount,
    cudaStream_t stream);

extern "C" void launchHizOcclusionCull(
    const OocBlockDepthInfo* d_depthInfo, unsigned int count,
    cudaTextureObject_t hizTexture, int hizWidth, int hizHeight,
    unsigned int* d_filteredBlockIds, unsigned int* d_filteredCount,
    cudaStream_t stream);

extern "C" void launchHizProbabilisticOcclusion(
    const OocBlockDepthInfo* d_depthInfo, unsigned int count,
    cudaTextureObject_t hizTexture, int hizWidth, int hizHeight,
    unsigned int* d_hizPassIndices, unsigned int* d_hizPassCount,
    unsigned int* d_filteredBlockIds, unsigned int* d_filteredCount,
    cudaStream_t stream);

extern "C" void launchHizDownsample(
    const unsigned int* d_depthBuffer, cudaSurfaceObject_t hizSurface,
    int fullWidth, int fullHeight, int hizWidth, int hizHeight,
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
    unsigned int* d_rasterAtomCount, cudaStream_t stream);

// ─────────────────── Thrust comparator ───────────────────

struct DepthInfoLess
{
    __host__ __device__
    bool operator()(const OocBlockDepthInfo& a, const OocBlockDepthInfo& b) const {
        return a.depth < b.depth;
    }
};

// ─────────────────── Pipeline GPU resources ───────────────────

struct OocGpuResources
{
    OocOctreeNode*    d_octree           = nullptr;
    unsigned int*     d_blockIndexBuffer = nullptr;
    OocBlockMetadata* d_blockMeta       = nullptr;
    unsigned int*     d_blockAtomCounts = nullptr;

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
    unsigned int* d_rasterAtomCount   = nullptr;

    unsigned int* d_hizPassIndices    = nullptr;
    unsigned int* d_hizPassCount      = nullptr;

    // Phase 3: LOD
    glm::vec4*    d_lodAtomBuffer     = nullptr;  ///< Always-resident LOD atoms
    glm::vec4*    d_lodActiveAtoms    = nullptr;  ///< LOD atoms selected this frame
    unsigned int* d_lodActiveCount    = nullptr;
    int           lodTotalAtoms       = 0;
    int           maxLodActiveAtoms   = 0;

    int maxBlocks     = 0;
    int maxOctreeNodes = 0;
    int maxActiveAtoms = 0;
};

static void allocGpuResources(OocGpuResources& r,
                               const ooc::PreprocessResult& prep,
                               int maxActiveAtoms,
                               int maxRequestsPerFrame)
{
    int numBlocks = static_cast<int>(prep.blocks.size());
    int numNodes  = static_cast<int>(prep.octreeNodes.size());
    r.maxBlocks      = numBlocks;
    r.maxOctreeNodes = numNodes;
    r.maxActiveAtoms = maxActiveAtoms;

    cudaMalloc(&r.d_octree,           numNodes  * sizeof(OocOctreeNode));
    cudaMalloc(&r.d_blockIndexBuffer, numBlocks * sizeof(unsigned int));
    cudaMalloc(&r.d_blockMeta,        numBlocks * sizeof(OocBlockMetadata));
    cudaMalloc(&r.d_blockAtomCounts,  numBlocks * sizeof(unsigned int));

    cudaMemcpy(r.d_octree, prep.octreeNodes.data(),
               numNodes * sizeof(OocOctreeNode), cudaMemcpyHostToDevice);
    cudaMemcpy(r.d_blockIndexBuffer, prep.blockIndexBuffer.data(),
               numBlocks * sizeof(unsigned int), cudaMemcpyHostToDevice);
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

    cudaMalloc(&r.d_requestBuffer, maxRequestsPerFrame * sizeof(unsigned int));
    cudaMalloc(&r.d_requestCount,  sizeof(unsigned int));

    cudaMalloc(&r.d_activeAtoms, maxActiveAtoms * sizeof(glm::vec4));
    cudaMalloc(&r.d_activeCount, sizeof(unsigned int));
    cudaMalloc(&r.d_rasterAtomCount, sizeof(unsigned int));

    cudaMalloc(&r.d_hizPassIndices, numBlocks * sizeof(unsigned int));
    cudaMalloc(&r.d_hizPassCount,   sizeof(unsigned int));

    // Phase 3: LOD buffer (always resident on GPU)
    r.lodTotalAtoms = static_cast<int>(prep.lodAtoms.size());
    if (r.lodTotalAtoms > 0) {
        cudaMalloc(&r.d_lodAtomBuffer, r.lodTotalAtoms * sizeof(glm::vec4));
        cudaMemcpy(r.d_lodAtomBuffer, prep.lodAtoms.data(),
                   r.lodTotalAtoms * sizeof(glm::vec4), cudaMemcpyHostToDevice);
    }
    // LOD active buffer: worst case all LOD atoms are visible
    r.maxLodActiveAtoms = std::max(r.lodTotalAtoms, 1);
    cudaMalloc(&r.d_lodActiveAtoms, r.maxLodActiveAtoms * sizeof(glm::vec4));
    cudaMalloc(&r.d_lodActiveCount, sizeof(unsigned int));
}

static void freeGpuResources(OocGpuResources& r) {
    cudaFree(r.d_octree);
    cudaFree(r.d_blockIndexBuffer);
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
    cudaFree(r.d_rasterAtomCount);
    cudaFree(r.d_hizPassIndices);
    cudaFree(r.d_hizPassCount);
    cudaFree(r.d_lodAtomBuffer);
    cudaFree(r.d_lodActiveAtoms);
    cudaFree(r.d_lodActiveCount);
}

/** Sum of cudaMalloc allocations in `allocGpuResources` (octree pipeline buffers). */
static size_t oocPipelineGpuDeviceBytes(const OocGpuResources& r, int maxRequestsPerFrame)
{
    const int numBlocks = r.maxBlocks;
    const int numNodes  = r.maxOctreeNodes;
    const int maxQueue  = numNodes + 8;
    size_t s = 0;
    s += static_cast<size_t>(numNodes) * sizeof(OocOctreeNode);
    s += static_cast<size_t>(numBlocks) * sizeof(unsigned int);  // d_blockIndexBuffer
    s += static_cast<size_t>(numBlocks) * sizeof(OocBlockMetadata);
    s += static_cast<size_t>(numBlocks) * sizeof(unsigned int);  // d_blockAtomCounts
    s += 2ull * static_cast<size_t>(maxQueue) * sizeof(unsigned int);
    s += 2ull * sizeof(unsigned int);
    s += static_cast<size_t>(numBlocks) * sizeof(unsigned int);
    s += sizeof(unsigned int);
    s += static_cast<size_t>(numBlocks) * sizeof(OocBlockDepthInfo);
    s += static_cast<size_t>(numBlocks) * sizeof(unsigned int);
    s += sizeof(unsigned int);
    s += static_cast<size_t>(maxRequestsPerFrame) * sizeof(unsigned int);
    s += sizeof(unsigned int);
    s += static_cast<size_t>(r.maxActiveAtoms) * sizeof(glm::vec4);
    s += sizeof(unsigned int);
    s += sizeof(unsigned int);  // d_rasterAtomCount
    s += static_cast<size_t>(numBlocks) * sizeof(unsigned int);
    s += sizeof(unsigned int);
    // LOD
    s += static_cast<size_t>(r.lodTotalAtoms) * sizeof(glm::vec4);
    s += static_cast<size_t>(r.maxLodActiveAtoms) * sizeof(glm::vec4);
    s += sizeof(unsigned int);
    return s;
}

static void appendOocPreprocessAndVramCsv(
    const std::string& csvPath,
    const SceneSettings& settings,
    int loadedAtomCount,
    const ooc::PreprocessResult& prep,
    size_t vramOocPipelineBytes,
    size_t vramStreamingBytes,
    size_t vramDepthBytes,
    size_t vramHizBytes,
    size_t vramColorEstBytes,
    size_t cudaMemFreeBytes,
    size_t cudaMemTotalBytes)
{
    if (csvPath.empty())
        return;

    std::filesystem::path p(csvPath);
    if (p.has_parent_path())
        std::filesystem::create_directories(p.parent_path());

    const bool needHeader = !std::filesystem::exists(p) ||
                            std::filesystem::file_size(p) == 0;

    std::ofstream out(csvPath, std::ios::app);
    if (!out) {
        std::cerr << "[OOC] preprocess stats: cannot open " << csvPath << std::endl;
        return;
    }

    if (needHeader) {
        out << "scene_type,sphere_count,block_count,octree_node_count,"
               "ms_morton_pipeline,ms_blocks_and_file,ms_octree_build,"
               "host_structures_bytes,block_file_bytes,"
               "vram_ooc_pipeline_bytes,vram_streaming_bytes,vram_depth_buffer_bytes,"
               "vram_hiz_bytes,vram_color_rgba8_est_bytes,vram_sum_estimated_bytes,"
               "cuda_mem_free_bytes,cuda_mem_total_bytes\n";
    }

    const auto& m = prep.metrics;
    const size_t vramSum = vramOocPipelineBytes + vramStreamingBytes + vramDepthBytes +
                           vramHizBytes + vramColorEstBytes;

    out << settings.sceneType << ','
        << loadedAtomCount << ','
        << m.blockCount << ','
        << m.octreeNodeCount << ','
        << m.msMortonPipeline << ','
        << m.msBlocksAndFile << ','
        << m.msOctreeBuild << ','
        << m.hostStructuresBytes << ','
        << m.blockFileBytes << ','
        << vramOocPipelineBytes << ','
        << vramStreamingBytes << ','
        << vramDepthBytes << ','
        << vramHizBytes << ','
        << vramColorEstBytes << ','
        << vramSum << ','
        << cudaMemFreeBytes << ','
        << cudaMemTotalBytes << '\n';
    out.close();
    std::cout << "[OOC] preprocess/VRAM metrics -> " << csvPath << std::endl;
}

// ═════════════════════════════════════════════════════════
// Batch statistics (CSV): accumulate N frames, write one row (append)
// ═════════════════════════════════════════════════════════

struct OocStatsBatchAccumulator {
    int                  batchSize   = 0;
    std::string          csvPath;
    std::string          sceneType;
    int                  sphereCount = 0;
    int                  totalBlocks = 0;
    int                  poolSlots   = 0;
    uint64_t             batchIndex   = 0;
    uint64_t             firstFrameId = 0;
    int                  framesInBatch = 0;

    uint64_t sumVisible   = 0;
    uint64_t sumFiltered  = 0;
    uint64_t sumRequests  = 0;
    uint64_t sumActive    = 0;
    uint32_t minVisible   = UINT32_MAX;
    uint32_t maxVisible   = 0;
    uint32_t minFiltered  = UINT32_MAX;
    uint32_t maxFiltered  = 0;
    uint32_t minRequests  = UINT32_MAX;
    uint32_t maxRequests  = 0;
    uint32_t minActive    = UINT32_MAX;
    uint32_t maxActive    = 0;

    double sumFrameMs     = 0.0;
    double minFrameMs     = std::numeric_limits<double>::max();
    double maxFrameMs     = 0.0;

    void resetBatchSums() {
        sumVisible = sumFiltered = sumRequests = sumActive = 0;
        minVisible = minFiltered = minRequests = minActive = UINT32_MAX;
        maxVisible = maxFiltered = maxRequests = maxActive = 0;
        sumFrameMs = 0.0;
        minFrameMs = std::numeric_limits<double>::max();
        maxFrameMs = 0.0;
        framesInBatch = 0;
    }

    void addFrame(uint32_t numVisible, uint32_t numFiltered, uint32_t numRequests,
                  uint32_t activeCount, double frameMs, uint64_t frameId)
    {
        if (batchSize <= 0) return;
        if (framesInBatch == 0)
            firstFrameId = frameId;

        sumVisible  += numVisible;
        sumFiltered += numFiltered;
        sumRequests += numRequests;
        sumActive   += activeCount;
        minVisible  = std::min(minVisible, numVisible);
        maxVisible  = std::max(maxVisible, numVisible);
        minFiltered = std::min(minFiltered, numFiltered);
        maxFiltered = std::max(maxFiltered, numFiltered);
        minRequests = std::min(minRequests, numRequests);
        maxRequests = std::max(maxRequests, numRequests);
        minActive   = std::min(minActive, activeCount);
        maxActive   = std::max(maxActive, activeCount);

        sumFrameMs += frameMs;
        minFrameMs  = std::min(minFrameMs, frameMs);
        maxFrameMs  = std::max(maxFrameMs, frameMs);
        framesInBatch++;
    }

    void writeCsvRow(int framesRecorded)
    {
        if (framesRecorded <= 0 || csvPath.empty()) return;

        std::filesystem::path p(csvPath);
        if (p.has_parent_path())
            std::filesystem::create_directories(p.parent_path());

        const bool needHeader = !std::filesystem::exists(p) ||
                                std::filesystem::file_size(p) == 0;

        std::ofstream out(csvPath, std::ios::app);
        if (!out) {
            std::cerr << "[OOC] stats: cannot open " << csvPath << std::endl;
            return;
        }

        if (needHeader) {
            out << "batch_index,first_frame,last_frame,frames,scene_type,sphere_count,total_blocks,pool_slots,"
                   "avg_numVisible,min_numVisible,max_numVisible,"
                   "avg_numFiltered,min_numFiltered,max_numFiltered,"
                   "avg_numRequests,min_numRequests,max_numRequests,"
                   "avg_activeCount,min_activeCount,max_activeCount,"
                   "avg_frame_ms,min_frame_ms,max_frame_ms\n";
        }

        double inv = 1.0 / static_cast<double>(framesRecorded);
        double avgVis = static_cast<double>(sumVisible) * inv;
        double avgFil = static_cast<double>(sumFiltered) * inv;
        double avgReq = static_cast<double>(sumRequests) * inv;
        double avgAct = static_cast<double>(sumActive) * inv;
        double avgMs  = sumFrameMs * inv;

        uint64_t lastFrame = firstFrameId + static_cast<uint64_t>(framesRecorded) - 1;

        out << batchIndex << ','
            << firstFrameId << ','
            << lastFrame << ','
            << framesRecorded << ','
            << sceneType << ','
            << sphereCount << ','
            << totalBlocks << ','
            << poolSlots << ','
            << avgVis << ',' << minVisible << ',' << maxVisible << ','
            << avgFil << ',' << minFiltered << ',' << maxFiltered << ','
            << avgReq << ',' << minRequests << ',' << maxRequests << ','
            << avgAct << ',' << minActive << ',' << maxActive << ','
            << avgMs << ',' << minFrameMs << ',' << maxFrameMs << '\n';

        out.close();
        batchIndex++;
    }

    void flushCompleteBatch()
    {
        if (batchSize <= 0 || framesInBatch < batchSize) return;
        writeCsvRow(framesInBatch);
        resetBatchSums();
    }

    /** Call on exit if the last batch is incomplete but should still be saved. */
    void flushPartialIfAny()
    {
        if (batchSize <= 0 || framesInBatch == 0) return;
        writeCsvRow(framesInBatch);
        resetBatchSums();
    }
};

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

    // Build OocConfig from JSON settings
    OocConfig oocCfg;
    oocCfg.atomsPerBlock       = settings.oocAtomsPerBlock;
    oocCfg.maxOctreeDepth      = settings.oocMaxOctreeDepth;
    oocCfg.blocksPerLeaf       = settings.oocBlocksPerLeaf;
    oocCfg.maxBlockPoolSlots   = settings.oocMaxBlockPoolSlots;
    oocCfg.maxRequestsPerFrame = settings.oocMaxRequestsPerFrame;
    oocCfg.visibilityThreshold = settings.oocVisibilityThreshold;
    oocCfg.occlusionMethod     = parseOcclusionMethod(settings.oocOcclusionMethod);

    const char* methodNames[] = {"NONE", "PROBABILISTIC", "PROBABILISTIC_OVERLAP", "HIZ", "HIZ_PROBABILISTIC"};
    std::cout << "[OOC] Config: atomsPerBlock=" << oocCfg.atomsPerBlock
              << " maxDepth=" << oocCfg.maxOctreeDepth
              << " blocksPerLeaf=" << oocCfg.blocksPerLeaf
              << " poolSlots=" << oocCfg.maxBlockPoolSlots
              << " maxReq=" << oocCfg.maxRequestsPerFrame
              << " visThreshold=" << oocCfg.visibilityThreshold
              << " occlusion=" << methodNames[static_cast<int>(oocCfg.occlusionMethod)]
              << std::endl;

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0)
    {
        std::cerr << "No CUDA devices found." << std::endl;
        return 1;
    }
    cudaSetDevice(0);

    // ── Load scene data on CPU ──
    std::vector<glm::vec4> spheres;
    std::vector<CylinderIndex> cylinders;
    getCompleteScene(spheres, sphereCount, cylinders, 0);
    const int actualAtomCount = static_cast<int>(spheres.size());
    std::cout << "Loaded " << actualAtomCount << " atoms." << std::endl;

    // ── Parse verbose flag ──
    bool octreeVerbose = false;
    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "-v" || arg == "--verbose") {
            octreeVerbose = true;
            std::cout << "[OOC] Verbose mode enabled." << std::endl;
            break;
        }
    }

    // ── Preprocess (once) ──
    ooc::PreprocessResult prep = ooc::preprocess(
        spheres.data(), static_cast<uint32_t>(actualAtomCount), "ooc_data", oocCfg, octreeVerbose);

    int totalBlocks = static_cast<int>(prep.blocks.size());
    int poolSlots   = std::min(oocCfg.maxBlockPoolSlots, totalBlocks);
    int maxActiveAtoms = poolSlots * oocCfg.atomsPerBlock;

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

    // ── HiZ for occlusion culling (previous frame depth) ──
    int hizLevel = settings.downsampleLevel;
    DepthDownsampleCUDA hizBuffer;
    bool useHiz = (oocCfg.occlusionMethod == OocOcclusionMethod::HIZ ||
                   oocCfg.occlusionMethod == OocOcclusionMethod::HIZ_PROBABILISTIC);
    if (useHiz) {
        hizBuffer.setup(hizLevel, screenWidth, screenHeight);
    }
    int hizWidth  = screenWidth  / (1 << hizLevel);
    int hizHeight = screenHeight / (1 << hizLevel);

    // ── GPU resources ──
    OocGpuResources gpu;
    allocGpuResources(gpu, prep, maxActiveAtoms, oocCfg.maxRequestsPerFrame);

    // ── Streaming manager ──
    ooc::StreamingManager streamMgr;
    streamMgr.initialize(poolSlots, totalBlocks,
                         prep.blockFilePath, prep.blocks,
                         oocCfg.atomsPerBlock, oocCfg.maxRequestsPerFrame);

    cudaDeviceSynchronize();
    size_t vramOocPipe = oocPipelineGpuDeviceBytes(gpu, oocCfg.maxRequestsPerFrame);
    size_t vramStream  = streamMgr.deviceMemoryBytes();
    size_t vramDepth =
        static_cast<size_t>(screenWidth) * static_cast<size_t>(screenHeight) *
        sizeof(unsigned int);
    size_t vramHiz = 0;
    if (useHiz && hizBuffer.isValid()) {
        vramHiz = static_cast<size_t>(hizWidth) * static_cast<size_t>(hizHeight) *
                  sizeof(float);
    }
    size_t vramColorEst = static_cast<size_t>(screenWidth) *
                          static_cast<size_t>(screenHeight) * 4ull;

    size_t cudaFreeMem = 0, cudaTotalMem = 0;
    cudaMemGetInfo(&cudaFreeMem, &cudaTotalMem);

    const size_t vramSumEst = vramOocPipe + vramStream + vramDepth + vramHiz + vramColorEst;
    std::cout << "[OOC] VRAM (estimated device buffers): pipeline=" << vramOocPipe
              << " B, streaming=" << vramStream << " B, depth=" << vramDepth
              << " B, HiZ=" << vramHiz << " B, color~RGBA8=" << vramColorEst
              << " B, sum=" << vramSumEst << " B" << std::endl;
    std::cout << "[OOC] cudaMemGetInfo: free=" << cudaFreeMem << " total=" << cudaTotalMem << std::endl;

    appendOocPreprocessAndVramCsv(
        settings.oocPreprocessStatsCsvPath, settings, actualAtomCount, prep,
        vramOocPipe, vramStream, vramDepth, vramHiz, vramColorEst, cudaFreeMem, cudaTotalMem);

    // ── Benchmark / profiler ──
    Benchmark benchmark(cameraController, checkpoints);
    int visibleAtoms = 0, drawnAtoms = 0;
    Profiler profiler(window, "media/csv/cuda_outofcore/frame_times.csv",
                      "media/csv/cuda_outofcore/process_times.csv",
                      sphereCount, 0, 0, settings.timerDuration);

    int oocMethodInt = static_cast<int>(oocCfg.occlusionMethod);
    std::unordered_map<std::string, int*> sceneData = {
        {"Screen width",   &screenWidth},
        {"Screen height",  &screenHeight},
        {"Total atoms",    &sphereCount},
        {"Visible atoms",  &visibleAtoms},
        {"Drawn atoms",    &drawnAtoms},
        {"Total blocks",   &totalBlocks},
        {"Pool slots",     &poolSlots},
        {"Occlusion method", &oocMethodInt}
    };
    float oocVisibilityThresholdUi = oocCfg.visibilityThreshold;
    window.setupSceneInfoGui("Scene Info", sceneData);
    window.setupCameraGui("Camera Info", &camera);
    window.setupInputInfoGui("Input Info");
    window.setupBenchmarkInfoGui("Benchmark", &benchmark);
    window.setupOocTuningGui("Out-of-core", oocVisibilityThresholdUi);

    // ── CUDA streams ──
    cudaStream_t renderStream, uploadStream;
    cudaStreamCreate(&renderStream);
    cudaStreamCreate(&uploadStream);

    // Pinned host buffers
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
                  oocCfg.maxRequestsPerFrame * sizeof(unsigned int),
                  cudaHostAllocDefault);
    cudaHostAlloc(&h_filteredBlockIds,
                  totalBlocks * sizeof(unsigned int),
                  cudaHostAllocDefault);

    uint64_t frameId = 0;

    // ── Phase 6: Temporal coherence state ──
    // When the camera is static (below movement thresholds), skip the
    // full culling pipeline and reuse the previous frame's filtered block list.
    glm::vec3 prevCamPos(0.0f);
    glm::vec3 prevCamFront(0.0f, 0.0f, -1.0f);
    bool hasPrevFrame = false;
    unsigned int cachedFilteredCount = 0;
    unsigned int* d_cachedFilteredBlockIds = nullptr;
    cudaMalloc(&d_cachedFilteredBlockIds,
               static_cast<size_t>(totalBlocks) * sizeof(unsigned int));
    static constexpr float CAMERA_POS_EPSILON   = 1e-4f;
    static constexpr float CAMERA_DOT_THRESHOLD = 0.99999f;

    OocStatsBatchAccumulator oocStats;
    oocStats.batchSize   = settings.oocStatsAccumulateFrames;
    oocStats.csvPath     = settings.oocStatsCsvPath;
    oocStats.sceneType   = settings.sceneType;
    oocStats.sphereCount = sphereCount;
    oocStats.totalBlocks = totalBlocks;
    oocStats.poolSlots   = poolSlots;

    // ── Render loop ──
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    bool isRunning = true;

    while (isRunning)
    {
        const auto frameT0 = std::chrono::high_resolution_clock::now();
        drawnAtoms = 0;
        cameraController.cameraUpdate();
        benchmark.update();
        profiler.updateProfiler(benchmark.getCheckpointID(),
                                visibleAtoms, drawnAtoms, 0, 0);

        nvtxRangePushA("OOC Frame");

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
        cst.visibilityThreshold = std::max(oocVisibilityThresholdUi, 1e-6f);
        cst.maxPoolSlots  = poolSlots;
        cst.atomsPerBlock = oocCfg.atomsPerBlock;
        cst.occlusionMethod = static_cast<int>(oocCfg.occlusionMethod);
        cst.nearPlane     = camera.getNear();
        cst.farPlane      = camera.getFar();
        cst.lodAreaThreshold = OOC_DEFAULT_LOD_AREA_THRESHOLD;
        cst.lodTotalAtoms    = gpu.lodTotalAtoms;

        cst.front = camera.getFront();
        cst.up    = camera.getUp();
        cst.right = camera.getRight();
        {
            float aspectRatio = static_cast<float>(screenWidth) / static_cast<float>(screenHeight);
            float fovRad = glm::radians(camera.getFov());
            float fovTan = tanf(fovRad * 0.5f);
            float halfFovTan = fovTan * aspectRatio;

            glm::vec3 camFront = camera.getFront();
            glm::vec3 camUp    = camera.getUp();
            glm::vec3 camRight = camera.getRight();

            glm::vec3 corner00 = glm::normalize(-halfFovTan * camRight - fovTan * camUp + camFront);
            glm::vec3 corner10 = glm::normalize( halfFovTan * camRight - fovTan * camUp + camFront);
            glm::vec3 corner01 = glm::normalize(-halfFovTan * camRight + fovTan * camUp + camFront);

            glm::mat3 viewRot = glm::mat3(camera.getView());
            glm::vec3 wCorner00 = viewRot * corner00;
            glm::vec3 wCorner10 = viewRot * corner10;
            glm::vec3 wCorner01 = viewRot * corner01;

            cst.rayStart = wCorner00;
            cst.dx = (wCorner10 - wCorner00) / static_cast<float>(screenWidth);
            cst.dy = (wCorner01 - wCorner00) / static_cast<float>(screenHeight);
        }

        cst.atomsColor = glm::vec3(0.8f, 0.1f, 0.1f);
        cst.diffuse    = 0.9f;

        nvtxRangePushA("Upload Constants");
        uploadOocConstants(cst, renderStream);
        uploadOocRasterConstants(cst, renderStream);
        nvtxRangePop();

        // ── Clear screen ──
        nvtxRangePushA("Screen Clear");
        launchScreenClearOoc(outputSurface, depthBuffer,
                             screenWidth, screenHeight,
                             camera.getFar(), renderStream);
        nvtxRangePop();

        // ── Phase 6: Temporal coherence — skip culling when camera is static ──
        glm::vec3 curCamPos   = camera.getPosition();
        glm::vec3 curCamFront = camera.getFront();
        bool cameraStatic = false;
        if (hasPrevFrame) {
            float dist = glm::length(curCamPos - prevCamPos);
            float dotF = glm::dot(glm::normalize(curCamFront), glm::normalize(prevCamFront));
            cameraStatic = (dist < CAMERA_POS_EPSILON && dotF > CAMERA_DOT_THRESHOLD);
        }

        unsigned int numVisible  = 0;
        unsigned int numFiltered = 0;
        unsigned int numRequests = 0;
        unsigned int activeCount = 0;

        if (cameraStatic && cachedFilteredCount > 0) {
            // ── Reuse previous frame's filtered block list ──
            nvtxRangePushA("Temporal Coherence (cache reuse)");
            numFiltered = cachedFilteredCount;
            cudaMemcpyAsync(gpu.d_filteredBlockIds, d_cachedFilteredBlockIds,
                            numFiltered * sizeof(unsigned int),
                            cudaMemcpyDeviceToDevice, renderStream);
            // Reset LOD counter — no BFS means no new LOD atoms
            cudaMemsetAsync(gpu.d_lodActiveCount, 0, sizeof(unsigned int), renderStream);
            visibleAtoms = static_cast<int>(numFiltered) * oocCfg.atomsPerBlock;
            nvtxRangePop();
        } else {
            // ── Full culling pipeline ──

            // Octree BFS frustum culling (LOD-aware — no per-level sync)
            nvtxRangePushA("Octree BFS Frustum Culling");
            cudaMemsetAsync(gpu.d_visibleBlockCount, 0, sizeof(unsigned int), renderStream);
            cudaMemsetAsync(gpu.d_lodActiveCount, 0, sizeof(unsigned int), renderStream);

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

            int maxQueueSize = gpu.maxOctreeNodes + 8;
            bool hasLod = (gpu.lodTotalAtoms > 0 && gpu.d_lodAtomBuffer != nullptr);
            for (int level = 0; level < oocCfg.maxOctreeDepth + 1; level++)
            {
                cudaMemsetAsync(outCount, 0, sizeof(unsigned int), renderStream);

                if (hasLod) {
                    launchOctreeFrustumCullLevelLod(
                        gpu.d_octree, gpu.d_blockIndexBuffer, inQueue, inCount,
                        outQueue, outCount, gpu.d_visibleBlockIds, gpu.d_visibleBlockCount,
                        gpu.maxBlocks, maxQueueSize,
                        gpu.d_lodAtomBuffer, gpu.d_lodActiveAtoms, gpu.d_lodActiveCount,
                        gpu.maxLodActiveAtoms, renderStream);
                } else {
                    launchOctreeFrustumCullLevelIndirect(
                        gpu.d_octree, gpu.d_blockIndexBuffer, inQueue, inCount,
                        outQueue, outCount, gpu.d_visibleBlockIds, gpu.d_visibleBlockCount,
                        gpu.maxBlocks, maxQueueSize, renderStream);
                }

                std::swap(inQueue, outQueue);
                std::swap(inCount, outCount);
            }

            cudaMemcpyAsync(h_visibleBlockCount, gpu.d_visibleBlockCount,
                            sizeof(unsigned int), cudaMemcpyDeviceToHost, renderStream);
            cudaStreamSynchronize(renderStream);
            nvtxRangePop(); // Octree BFS Frustum Culling
            numVisible = *h_visibleBlockCount;
            visibleAtoms = static_cast<int>(numVisible) * oocCfg.atomsPerBlock;

            if (numVisible > 0)
            {
                // Depth + area, sort, occlusion culling
                nvtxRangePushA("Compute Block Depth+Area");
                launchComputeBlockDepthArea(
                    gpu.d_blockMeta, gpu.d_visibleBlockIds, numVisible,
                    gpu.d_depthInfo, renderStream);
                nvtxRangePop();

                nvtxRangePushA("Thrust Sort (Depth)");
                thrust::device_ptr<OocBlockDepthInfo> depthPtr(gpu.d_depthInfo);
                thrust::sort(thrust::cuda::par.on(renderStream),
                             depthPtr, depthPtr + numVisible, DepthInfoLess());
                nvtxRangePop();

                cudaMemsetAsync(gpu.d_filteredCount, 0, sizeof(unsigned int), renderStream);

                nvtxRangePushA("Occlusion Culling");
                switch (oocCfg.occlusionMethod) {
                    case OocOcclusionMethod::NONE:
                        launchNoOcclusionPassthrough(
                            gpu.d_depthInfo, numVisible,
                            gpu.d_filteredBlockIds, gpu.d_filteredCount, renderStream);
                        break;

                    case OocOcclusionMethod::PROBABILISTIC:
                        launchProbabilisticOcclusion(
                            gpu.d_depthInfo, numVisible,
                            gpu.d_filteredBlockIds, gpu.d_filteredCount, renderStream);
                        break;

                    case OocOcclusionMethod::PROBABILISTIC_OVERLAP:
                        launchProbabilisticOcclusionOverlap(
                            gpu.d_depthInfo, numVisible,
                            gpu.d_filteredBlockIds, gpu.d_filteredCount, renderStream);
                        break;

                    case OocOcclusionMethod::HIZ:
                        if (useHiz && hizBuffer.isValid()) {
                            launchHizOcclusionCull(
                                gpu.d_depthInfo, numVisible,
                                hizBuffer.getTexture(), hizWidth, hizHeight,
                                gpu.d_filteredBlockIds, gpu.d_filteredCount, renderStream);
                        } else {
                            launchNoOcclusionPassthrough(
                                gpu.d_depthInfo, numVisible,
                                gpu.d_filteredBlockIds, gpu.d_filteredCount, renderStream);
                        }
                        break;

                    case OocOcclusionMethod::HIZ_PROBABILISTIC:
                        if (useHiz && hizBuffer.isValid()) {
                            launchHizProbabilisticOcclusion(
                                gpu.d_depthInfo, numVisible,
                                hizBuffer.getTexture(), hizWidth, hizHeight,
                                gpu.d_hizPassIndices, gpu.d_hizPassCount,
                                gpu.d_filteredBlockIds, gpu.d_filteredCount, renderStream);
                        } else {
                            launchProbabilisticOcclusion(
                                gpu.d_depthInfo, numVisible,
                                gpu.d_filteredBlockIds, gpu.d_filteredCount, renderStream);
                        }
                        break;
                }
                nvtxRangePop(); // Occlusion Culling

                cudaMemcpyAsync(h_filteredCount, gpu.d_filteredCount,
                                sizeof(unsigned int), cudaMemcpyDeviceToHost, renderStream);
                cudaStreamSynchronize(renderStream);
                numFiltered = std::min(*h_filteredCount,
                                       static_cast<unsigned int>(poolSlots));
            }

            // Cache filtered list for temporal coherence
            if (numFiltered > 0) {
                cudaMemcpyAsync(d_cachedFilteredBlockIds, gpu.d_filteredBlockIds,
                                numFiltered * sizeof(unsigned int),
                                cudaMemcpyDeviceToDevice, renderStream);
            }
            cachedFilteredCount = numFiltered;
        } // end else (full culling pipeline)

        // ── Request generation, streaming, render (always when numFiltered > 0) ──
        if (numFiltered > 0)
        {
            // Request generation
            nvtxRangePushA("Request Generation");
            cudaMemsetAsync(gpu.d_requestCount, 0, sizeof(unsigned int), renderStream);
            launchComputeBlockRequests(
                gpu.d_filteredBlockIds, numFiltered,
                streamMgr.getReadSlotMap(),
                gpu.d_requestBuffer, gpu.d_requestCount,
                oocCfg.maxRequestsPerFrame, renderStream);

            cudaMemcpyAsync(h_requestCount, gpu.d_requestCount,
                            sizeof(unsigned int), cudaMemcpyDeviceToHost, renderStream);
            cudaMemcpyAsync(h_filteredBlockIds, gpu.d_filteredBlockIds,
                            numFiltered * sizeof(unsigned int),
                            cudaMemcpyDeviceToHost, renderStream);
            cudaStreamSynchronize(renderStream);
            numRequests = std::min(*h_requestCount,
                static_cast<unsigned int>(oocCfg.maxRequestsPerFrame));

            if (numRequests > 0)
            {
                nvtxRangePushA("Thrust Sort+Unique (Requests)");
                thrust::device_ptr<unsigned int> reqPtr(gpu.d_requestBuffer);
                thrust::sort(thrust::cuda::par.on(renderStream),
                             reqPtr, reqPtr + numRequests);
                auto newEnd = thrust::unique(thrust::cuda::par.on(renderStream),
                                            reqPtr, reqPtr + numRequests);
                numRequests = static_cast<unsigned int>(newEnd - reqPtr);
                nvtxRangePop();

                cudaMemcpyAsync(h_requestBuffer, gpu.d_requestBuffer,
                                numRequests * sizeof(unsigned int),
                                cudaMemcpyDeviceToHost, renderStream);
                cudaStreamSynchronize(renderStream);
            }
            nvtxRangePop(); // Request Generation

            // CPU streaming (uploads go to write buffer, async)
            nvtxRangePushA("CPU Streaming");
            streamMgr.processRequests(
                h_requestBuffer, numRequests,
                h_filteredBlockIds, numFiltered,
                frameId, uploadStream);
            nvtxRangePop();

            // Build active atom list + render
            nvtxRangePushA("Build Active Atom List");
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
            nvtxRangePop(); // Build Active Atom List
            activeCount = *h_activeCount;

            if (activeCount > 0) {
                nvtxRangePushA("Sphere Raster OOC");
                cudaMemsetAsync(gpu.d_rasterAtomCount, 0, sizeof(unsigned int),
                                renderStream);
                launchSphereRasterOoc(
                    gpu.d_activeAtoms, activeCount,
                    depthBuffer, outputSurface,
                    gpu.d_rasterAtomCount, renderStream);
                nvtxRangePop();
            }
        }

        // Phase 3: Rasterize LOD atoms (always, independent of streaming)
        {
            unsigned int lodCount = 0;
            cudaMemcpyAsync(&lodCount, gpu.d_lodActiveCount,
                            sizeof(unsigned int), cudaMemcpyDeviceToHost, renderStream);
            cudaStreamSynchronize(renderStream);
            if (lodCount > 0) {
                nvtxRangePushA("LOD Sphere Raster");
                launchSphereRasterOoc(
                    gpu.d_lodActiveAtoms, lodCount,
                    depthBuffer, outputSurface,
                    gpu.d_rasterAtomCount, renderStream);
                drawnAtoms += static_cast<int>(lodCount);
                nvtxRangePop();
            }
        }

        // Update camera state for next frame's temporal coherence check
        prevCamPos   = curCamPos;
        prevCamFront = curCamFront;
        hasPrevFrame = true;

        // ── HiZ update (for next frame) ──
        if (useHiz && hizBuffer.isValid()) {
            nvtxRangePushA("HiZ Downsample");
            launchHizDownsample(depthBuffer, hizBuffer.getSurface(),
                                screenWidth, screenHeight,
                                hizWidth, hizHeight, renderStream);
            nvtxRangePop();
        }

        if (frameId % 60 == 0 && octreeVerbose)
        {
            std::cout << "[OOC] Frame " << frameId
                      << " visible=" << numVisible << " filtered=" << numFiltered
                      << " requests=" << numRequests << " active_list=" << activeCount
                      << " raster_atoms=" << drawnAtoms
                      << (cameraStatic ? " [CACHED]" : "") << std::endl;
        }

        // Phase 2: deferred raster count readback (moved from raster section)
        if (activeCount > 0) {
            unsigned int rasterSubmitted = 0;
            cudaMemcpyAsync(&rasterSubmitted, gpu.d_rasterAtomCount,
                            sizeof(unsigned int), cudaMemcpyDeviceToHost, renderStream);
            cudaStreamSynchronize(renderStream);
            drawnAtoms = static_cast<int>(rasterSubmitted);
        }

        // Ensure upload stream finished before swapping buffers
        nvtxRangePushA("Swap Buffers");
        cudaStreamSynchronize(uploadStream);
        streamMgr.swapBuffers();
        nvtxRangePop();

        nvtxRangePop(); // OOC Frame

        glBindFramebuffer(GL_READ_FRAMEBUFFER, canvas.getFramebuffer().getId());
        isRunning = window.update();
        glFinish();

        const auto frameT1 = std::chrono::high_resolution_clock::now();
        const double frameMs =
            std::chrono::duration<double, std::milli>(frameT1 - frameT0).count();
        oocStats.addFrame(numVisible, numFiltered, numRequests,
                          static_cast<uint32_t>(std::max(0, drawnAtoms)), frameMs, frameId);
        oocStats.flushCompleteBatch();
        frameId++;
    }

    oocStats.flushPartialIfAny();

    // ── Cleanup ──
    cudaStreamDestroy(renderStream);
    cudaStreamDestroy(uploadStream);
    texWrapper.cudaUnmapResources();
    texWrapper.cudaDestroySurfaceObj();
    freeGpuResources(gpu);
    streamMgr.destroy();

    // Phase 6: free cached filtered list
    cudaFree(d_cachedFilteredBlockIds);

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
