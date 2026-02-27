/**
 * @file outofcore_kernels.cu
 * @brief GPU kernels for out-of-core octree traversal, probabilistic
 *        occlusion culling, request generation, and active atom list
 *        construction.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>

#include "outofcore/outofcore_types.h"

// ─────────────────── Constant memory ───────────────────

static __constant__ OocConstants oocCst;

extern "C" void uploadOocConstants(const OocConstants& constants, cudaStream_t stream) {
    cudaMemcpyToSymbolAsync(oocCst, &constants, sizeof(OocConstants), 0,
                            cudaMemcpyHostToDevice, stream);
}

// ─────────────────── AABB–frustum test ───────────────────

__device__ inline bool aabbInsideFrustum(glm::vec3 bmin, glm::vec3 bmax) {
    for (int i = 0; i < 6; i++) {
        glm::vec3 n(oocCst.frustumPlanes[i]);
        float d = oocCst.frustumPlanes[i].w;
        glm::vec3 p;
        p.x = (n.x >= 0.0f) ? bmax.x : bmin.x;
        p.y = (n.y >= 0.0f) ? bmax.y : bmin.y;
        p.z = (n.z >= 0.0f) ? bmax.z : bmin.z;
        if (glm::dot(n, p) - d < 0.0f) return false;
    }
    return true;
}

// ═════════════════════════════════════════════════════════
// Kernel 1: Octree frustum culling — BFS by levels
// ═════════════════════════════════════════════════════════

/**
 * Each thread processes one node from the input queue. If visible and
 * interior, its children are appended to the output queue. If visible
 * and leaf, its block range is appended to the visible-blocks list.
 */
__global__ void octreeFrustumCullLevelKernel(
    const OocOctreeNode* __restrict__ octree,
    const unsigned int*  __restrict__ inQueue,
    unsigned int                      inCount,
    unsigned int*        __restrict__ outQueue,
    unsigned int*        __restrict__ outCount,
    unsigned int*        __restrict__ visibleBlockIds,
    unsigned int*        __restrict__ visibleBlockCount,
    int                               maxVisibleBlocks)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= inCount) return;

    unsigned int nodeIdx = inQueue[idx];
    OocOctreeNode node   = octree[nodeIdx];

    if (!aabbInsideFrustum(node.aabbMin, node.aabbMax)) return;

    if (node.childBaseIndex < 0) 
    {
        // Leaf — emit block IDs
        for (int b = node.blockRangeStart; b < node.blockRangeEnd; b++) 
        {
            unsigned int out = atomicAdd(visibleBlockCount, 1u);
            if (out < static_cast<unsigned int>(maxVisibleBlocks))
                visibleBlockIds[out] = static_cast<unsigned int>(b);
        }
    } else 
    {
        // Interior — enqueue children
        int childBase = node.childBaseIndex;
        int childIdx  = 0;
        for (int oct = 0; oct < 8; oct++) 
        {
            if (node.childMask & (1u << oct)) 
            {
                unsigned int out = atomicAdd(outCount, 1u);
                outQueue[out] = static_cast<unsigned int>(childBase + childIdx);
                childIdx++;
            }
        }
    }
}

extern "C" void launchOctreeFrustumCullLevel(
    const OocOctreeNode* d_octree,
    const unsigned int*  d_inQueue,
    unsigned int         inCount,
    unsigned int*        d_outQueue,
    unsigned int*        d_outCount,
    unsigned int*        d_visibleBlockIds,
    unsigned int*        d_visibleBlockCount,
    int                  maxVisibleBlocks,
    cudaStream_t         stream)
{
    if (inCount == 0) return;
    dim3 block(256);
    dim3 grid((inCount + block.x - 1) / block.x);
    octreeFrustumCullLevelKernel<<<grid, block, 0, stream>>>(
        d_octree, d_inQueue, inCount, d_outQueue, d_outCount,
        d_visibleBlockIds, d_visibleBlockCount, maxVisibleBlocks);
}

// ═════════════════════════════════════════════════════════
// Kernel 2: Compute block depth and projected area
// ═════════════════════════════════════════════════════════

__device__ inline glm::vec3 aabbCorner(glm::vec3 bmin, glm::vec3 bmax, int i) {
    return {
        (i & 1) ? bmax.x : bmin.x,
        (i & 2) ? bmax.y : bmin.y,
        (i & 4) ? bmax.z : bmin.z
    };
}

__global__ void computeBlockDepthAreaKernel(
    const OocBlockMetadata* __restrict__ blockMeta,
    const unsigned int*     __restrict__ visibleBlockIds,
    unsigned int                         visibleCount,
    OocBlockDepthInfo*      __restrict__ depthInfo)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= visibleCount) return;

    unsigned int bid = visibleBlockIds[idx];
    OocBlockMetadata bm = blockMeta[bid];

    float depth = glm::length(bm.center - oocCst.cameraPos);

    glm::vec2 smin(1e6f), smax(-1e6f);
    for (int i = 0; i < 8; i++) 
    {
        glm::vec3 corner = aabbCorner(bm.aabbMin, bm.aabbMax, i);
        glm::vec4 clip = oocCst.viewProj * glm::vec4(corner, 1.0f);
        if (clip.w > 0.001f) 
        {
            float iw = 1.0f / clip.w;
            float sx = (clip.x * iw * 0.5f + 0.5f) * static_cast<float>(oocCst.screenWidth);
            float sy = (clip.y * iw * 0.5f + 0.5f) * static_cast<float>(oocCst.screenHeight);
            smin.x = fminf(smin.x, sx);
            smin.y = fminf(smin.y, sy);
            smax.x = fmaxf(smax.x, sx);
            smax.y = fmaxf(smax.y, sy);
        }
    }

    float area = fmaxf(0.0f, smax.x - smin.x) * fmaxf(0.0f, smax.y - smin.y);

    OocBlockDepthInfo info;
    info.blockId       = bid;
    info.depth         = depth;
    info.projectedArea = area;
    info._pad          = 0.0f;
    depthInfo[idx]     = info;
}

extern "C" void launchComputeBlockDepthArea(
    const OocBlockMetadata* d_blockMeta,
    const unsigned int*     d_visibleBlockIds,
    unsigned int            visibleCount,
    OocBlockDepthInfo*      d_depthInfo,
    cudaStream_t            stream)
{
    if (visibleCount == 0) return;
    dim3 block(256);
    dim3 grid((visibleCount + block.x - 1) / block.x);
    computeBlockDepthAreaKernel<<<grid, block, 0, stream>>>(
        d_blockMeta, d_visibleBlockIds, visibleCount, d_depthInfo);
}

// ═════════════════════════════════════════════════════════
// Kernel 3: Probabilistic occlusion culling
// ═════════════════════════════════════════════════════════

/**
 * Runs on a single thread because the recurrence v_c = (1-D_c)*v_{c-1}
 * is inherently sequential. Operates on the sorted block list (already
 * ordered front-to-back by thrust::sort).
 */
__global__ void probabilisticOcclusionKernel(
    const OocBlockDepthInfo* __restrict__ sorted,
    unsigned int                          count,
    unsigned int*            __restrict__ filteredBlockIds,
    unsigned int*            __restrict__ filteredCount)
{
    float totalArea = static_cast<float>(oocCst.screenWidth) *
                      static_cast<float>(oocCst.screenHeight);
    float v = 1.0f;

    for (unsigned int c = 0; c < count; c++) 
    {
        float Dc = sorted[c].projectedArea / totalArea;
        Dc = fminf(fmaxf(Dc, 0.0f), 1.0f);
        v *= (1.0f - Dc);

        if (v > oocCst.visibilityThreshold) 
        {
            unsigned int out = atomicAdd(filteredCount, 1u);
            filteredBlockIds[out] = sorted[c].blockId;
        }
    }
}

extern "C" void launchProbabilisticOcclusion(
    const OocBlockDepthInfo* d_sorted,
    unsigned int             count,
    unsigned int*            d_filteredBlockIds,
    unsigned int*            d_filteredCount,
    cudaStream_t             stream)
{
    if (count == 0) return;
    probabilisticOcclusionKernel<<<1, 1, 0, stream>>>(
        d_sorted, count, d_filteredBlockIds, d_filteredCount);
}

// ═════════════════════════════════════════════════════════
// Kernel 3b: Probabilistic occlusion with dynamic parallelism
// ═════════════════════════════════════════════════════════
//
// Alternative implementation: parent kernel launches child kernels per chunk.
// The parent first precomputes v_start for each chunk (sequential loop), then
// launches all children. Children are independent and can run in parallel.
// No cudaDeviceSynchronize in device code (host-only API).
// ═════════════════════════════════════════════════════════

static constexpr unsigned int OOC_OCCLUSION_CHUNK_SIZE = 64;

__global__ void probabilisticOcclusionChildKernel(
    const OocBlockDepthInfo* __restrict__ sorted,
    unsigned int             start,
    unsigned int             end,
    float                    vStart,
    float                    totalArea,
    float                    threshold,
    unsigned int*            __restrict__ filteredBlockIds,
    unsigned int*            __restrict__ filteredCount)
{
    float v = vStart;
    for (unsigned int c = start; c < end; c++) {
        float Dc = sorted[c].projectedArea / totalArea;
        Dc = fminf(fmaxf(Dc, 0.0f), 1.0f);
        v *= (1.0f - Dc);

        if (v > threshold) {
            unsigned int out = atomicAdd(filteredCount, 1u);
            filteredBlockIds[out] = sorted[c].blockId;
        }
    }
}

__global__ void probabilisticOcclusionParentKernel(
    const OocBlockDepthInfo* __restrict__ sorted,
    unsigned int             count,
    unsigned int*            __restrict__ filteredBlockIds,
    unsigned int*            __restrict__ filteredCount,
    float*                   __restrict__ d_vStartArray)
{
    float totalArea = static_cast<float>(oocCst.screenWidth) *
                      static_cast<float>(oocCst.screenHeight);
    float threshold = oocCst.visibilityThreshold;

    // Phase 1: precompute v_start for each chunk (sequential)
    unsigned int numChunks = (count + OOC_OCCLUSION_CHUNK_SIZE - 1) / OOC_OCCLUSION_CHUNK_SIZE;
    float v = 1.0f;
    for (unsigned int k = 0; k < numChunks; k++) {
        d_vStartArray[k] = v;
        unsigned int chunkStart = k * OOC_OCCLUSION_CHUNK_SIZE;
        unsigned int chunkEnd   = min(chunkStart + OOC_OCCLUSION_CHUNK_SIZE, count);
        for (unsigned int c = chunkStart; c < chunkEnd; c++) {
            float Dc = sorted[c].projectedArea / totalArea;
            Dc = fminf(fmaxf(Dc, 0.0f), 1.0f);
            v *= (1.0f - Dc);
        }
    }

    // Phase 2: launch child kernels (all independent, no sync needed)
    for (unsigned int k = 0; k < numChunks; k++) {
        unsigned int chunkStart = k * OOC_OCCLUSION_CHUNK_SIZE;
        unsigned int chunkEnd   = min(chunkStart + OOC_OCCLUSION_CHUNK_SIZE, count);
        if (chunkStart >= chunkEnd) break;

        probabilisticOcclusionChildKernel<<<1, 1>>>(
            sorted, chunkStart, chunkEnd, d_vStartArray[k],
            totalArea, threshold, filteredBlockIds, filteredCount);
    }
}

extern "C" void launchProbabilisticOcclusionDynamicParallelism(
    const OocBlockDepthInfo* d_sorted,
    unsigned int             count,
    unsigned int*            d_filteredBlockIds,
    unsigned int*            d_filteredCount,
    float*                   d_vEndBuffer,
    cudaStream_t             stream)
{
    if (count == 0) return;
    probabilisticOcclusionParentKernel<<<1, 1, 0, stream>>>(
        d_sorted, count, d_filteredBlockIds, d_filteredCount, d_vEndBuffer);
}

// ═════════════════════════════════════════════════════════
// Kernel 4: Compute block requests (missing from pool)
// ═════════════════════════════════════════════════════════

__global__ void computeBlockRequestsKernel(
    const unsigned int* __restrict__ filteredBlockIds,
    unsigned int                     filteredCount,
    const int32_t*      __restrict__ blockSlotMap,
    unsigned int*       __restrict__ requestBuffer,
    unsigned int*       __restrict__ requestCount,
    int                              maxRequests)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= filteredCount) return;

    uint32_t bid = filteredBlockIds[idx];
    if (blockSlotMap[bid] < 0) 
    {
        unsigned int out = atomicAdd(requestCount, 1u);
        if (out < static_cast<unsigned int>(maxRequests))
            requestBuffer[out] = bid;
    }
}

extern "C" void launchComputeBlockRequests(
    const unsigned int* d_filteredBlockIds,
    unsigned int        filteredCount,
    const int32_t*      d_blockSlotMap,
    unsigned int*       d_requestBuffer,
    unsigned int*       d_requestCount,
    int                 maxRequests,
    cudaStream_t        stream)
{
    if (filteredCount == 0) return;
    dim3 block(256);
    dim3 grid((filteredCount + block.x - 1) / block.x);
    computeBlockRequestsKernel<<<grid, block, 0, stream>>>(
        d_filteredBlockIds, filteredCount, d_blockSlotMap,
        d_requestBuffer, d_requestCount, maxRequests);
}

// ═════════════════════════════════════════════════════════
// Kernel 5: Build active atom list from loaded pool slots
// ═════════════════════════════════════════════════════════

__global__ void buildActiveAtomListKernel(
    const glm::vec4*    __restrict__ atomPool,
    const int32_t*      __restrict__ blockSlotMap,
    const unsigned int* __restrict__ filteredBlockIds,
    unsigned int                     filteredCount,
    const unsigned int* __restrict__ blockAtomCounts,
    glm::vec4*          __restrict__ activeAtoms,
    unsigned int*       __restrict__ activeCount)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= filteredCount) return;

    uint32_t bid = filteredBlockIds[idx];
    int32_t slot = blockSlotMap[bid];
    if (slot < 0) return;

    unsigned int count = blockAtomCounts[bid];
    unsigned int base  = atomicAdd(activeCount, count);

    const glm::vec4* src = atomPool + static_cast<size_t>(slot) * OOC_ATOMS_PER_BLOCK;
    for (unsigned int i = 0; i < count; i++) {
        activeAtoms[base + i] = src[i];
    }
}

extern "C" void launchBuildActiveAtomList(
    const glm::vec4*    d_atomPool,
    const int32_t*      d_blockSlotMap,
    const unsigned int* d_filteredBlockIds,
    unsigned int        filteredCount,
    const unsigned int* d_blockAtomCounts,
    glm::vec4*          d_activeAtoms,
    unsigned int*       d_activeCount,
    cudaStream_t        stream)
{
    if (filteredCount == 0) return;
    dim3 block(256);
    dim3 grid((filteredCount + block.x - 1) / block.x);
    buildActiveAtomListKernel<<<grid, block, 0, stream>>>(
        d_atomPool, d_blockSlotMap, d_filteredBlockIds, filteredCount,
        d_blockAtomCounts, d_activeAtoms, d_activeCount);
}
