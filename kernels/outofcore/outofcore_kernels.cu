/**
 * @file outofcore_kernels.cu
 * @brief GPU kernels for out-of-core octree traversal, occlusion culling
 *        (multiple methods), request generation, and active atom list
 *        construction.
 *
 * Occlusion culling methods (selected at runtime via OocConstants.occlusionMethod):
 *   0 = NONE                  — all visible blocks pass through
 *   1 = PROBABILISTIC         — Atomsviewer with real density Dc
 *   2 = PROBABILISTIC_OVERLAP — same but only accumulates v when blocks overlap
 *   3 = HIZ                   — two-pass style HiZ test against previous frame
 *   4 = HIZ_PROBABILISTIC     — HiZ (parallel) then probabilistic (sequential)
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

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

__global__ void octreeFrustumCullLevelKernel(
    const OocOctreeNode* __restrict__ octree,
    const unsigned int*  __restrict__ blockIndexBuffer,
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
        for (int i = node.blockRangeStart; i < node.blockRangeEnd; i++)
        {
            unsigned int bid = blockIndexBuffer[i];
            unsigned int out = atomicAdd(visibleBlockCount, 1u);
            if (out < static_cast<unsigned int>(maxVisibleBlocks))
                visibleBlockIds[out] = bid;
        }
    } else
    {
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
    const unsigned int*  d_blockIndexBuffer,
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
        d_octree, d_blockIndexBuffer, d_inQueue, inCount, d_outQueue, d_outCount,
        d_visibleBlockIds, d_visibleBlockCount, maxVisibleBlocks);
}

// ═════════════════════════════════════════════════════════
// Kernel 2: Compute block depth, projected AABB rect, real density,
//           and min NDC depth for HiZ
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

    float sw = static_cast<float>(oocCst.screenWidth);
    float sh = static_cast<float>(oocCst.screenHeight);
    glm::vec2 smin(sw, sh), smax(0.0f, 0.0f);

    // Track minimum NDC depth across all 8 AABB corners
    float minNdc = 1.0f;
    int validCorners = 0;

    for (int i = 0; i < 8; i++)
    {
        glm::vec3 corner = aabbCorner(bm.aabbMin, bm.aabbMax, i);
        glm::vec4 clip = oocCst.viewProj * glm::vec4(corner, 1.0f);
        if (clip.w > 0.1f)
        {
            float iw = 1.0f / clip.w;
            float sx = fminf(fmaxf((clip.x * iw * 0.5f + 0.5f) * sw, 0.0f), sw);
            float sy = fminf(fmaxf((clip.y * iw * 0.5f + 0.5f) * sh, 0.0f), sh);
            smin.x = fminf(smin.x, sx);
            smin.y = fminf(smin.y, sy);
            smax.x = fmaxf(smax.x, sx);
            smax.y = fmaxf(smax.y, sy);

            // NDC depth = clip.z / clip.w (same depth the rasterizer computes)
            float ndcZ = clip.z * iw;
            minNdc = fminf(minNdc, ndcZ);
            validCorners++;
        }
    }

    float aabbArea = fmaxf(0.0f, smax.x - smin.x) * fmaxf(0.0f, smax.y - smin.y);
    aabbArea = fminf(aabbArea, sw * sh);

    float fovFactor = sh / (2.0f * tanf(oocCst.fov * 0.5f * 3.14159265f / 180.0f));
    float avgRadius = 0.5f;
    float rProj = (avgRadius / fmaxf(depth, 0.01f)) * fovFactor;
    float singleAtomArea = 3.14159265f * rProj * rProj;
    float realCoverage = static_cast<float>(bm.atomCount) * singleAtomArea;
    float projectedArea = fminf(realCoverage, aabbArea);

    OocBlockDepthInfo info;
    info.blockId       = bid;
    info.depth         = depth;
    info.projectedArea = projectedArea;
    info.screenMinX    = smin.x;
    info.screenMinY    = smin.y;
    info.screenMaxX    = smax.x;
    info.screenMaxY    = smax.y;
    info.atomCount     = bm.atomCount;
    info.minNdcDepth   = (validCorners > 0) ? minNdc : 0.0f;
    info.screenMinU    = smin.x / sw;
    info.screenMinV    = smin.y / sh;
    info.screenMaxU    = smax.x / sw;
    info.screenMaxV    = smax.y / sh;
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
// Kernel 3a: Probabilistic occlusion with real density
// ═════════════════════════════════════════════════════════

/* --- OLD VERSION (AABB-based Dc, no overlap check) ---
 * __global__ void probabilisticOcclusionKernel_OLD(...)
 * {
 *     float v = 1.0f;
 *     for (unsigned int c = 0; c < count; c++) {
 *         float Dc = sorted[c].projectedArea / totalArea;
 *         Dc = fminf(fmaxf(Dc, 0.0f), 1.0f);
 *         v *= (1.0f - Dc);
 *         if (v > oocCst.visibilityThreshold) {
 *             atomicAdd(filteredCount, 1u); ...
 *         }
 *     }
 * }
 * --- END OLD VERSION --- */

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
        unsigned int out = atomicAdd(filteredCount, 1u);
        filteredBlockIds[out] = sorted[c].blockId;

        float Dc = sorted[c].projectedArea / totalArea;
        Dc = fminf(fmaxf(Dc, 0.0f), 1.0f);
        v *= (1.0f - Dc);

        if (v <= oocCst.visibilityThreshold) break;
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
// Kernel 3b: Probabilistic occlusion WITH overlap check
// ═════════════════════════════════════════════════════════

__device__ inline bool rectsOverlap(float ax0, float ay0, float ax1, float ay1,
                                     float bx0, float by0, float bx1, float by1) {
    return !(ax1 <= bx0 || bx1 <= ax0 || ay1 <= by0 || by1 <= ay0);
}

__global__ void probabilisticOcclusionOverlapKernel(
    const OocBlockDepthInfo* __restrict__ sorted,
    unsigned int                          count,
    unsigned int*            __restrict__ filteredBlockIds,
    unsigned int*            __restrict__ filteredCount)
{
    float totalArea = static_cast<float>(oocCst.screenWidth) *
                      static_cast<float>(oocCst.screenHeight);

    for (unsigned int i = 0; i < count; i++)
    {
        float v = 1.0f;
        for (unsigned int j = 0; j < i; j++)
        {
            if (!rectsOverlap(sorted[i].screenMinX, sorted[i].screenMinY,
                              sorted[i].screenMaxX, sorted[i].screenMaxY,
                              sorted[j].screenMinX, sorted[j].screenMinY,
                              sorted[j].screenMaxX, sorted[j].screenMaxY))
                continue;

            float Dc = sorted[j].projectedArea / totalArea;
            Dc = fminf(fmaxf(Dc, 0.0f), 1.0f);
            v *= (1.0f - Dc);
        }

        if (v > oocCst.visibilityThreshold)
        {
            unsigned int out = atomicAdd(filteredCount, 1u);
            filteredBlockIds[out] = sorted[i].blockId;
        }
    }
}

extern "C" void launchProbabilisticOcclusionOverlap(
    const OocBlockDepthInfo* d_sorted,
    unsigned int             count,
    unsigned int*            d_filteredBlockIds,
    unsigned int*            d_filteredCount,
    cudaStream_t             stream)
{
    if (count == 0) return;
    probabilisticOcclusionOverlapKernel<<<1, 1, 0, stream>>>(
        d_sorted, count, d_filteredBlockIds, d_filteredCount);
}

// ═════════════════════════════════════════════════════════
// Kernel 3c: HiZ occlusion culling (per-block, parallelizable)
// ═════════════════════════════════════════════════════════
//
// State-of-the-art two-pass style HiZ test:
//
//   1. For each block, compute the screen-space UV rect and find the
//      correct mip level where the AABB covers ≤ 4 texels.
//      (Since we use a single-level downsample, we pick the level
//       where the AABB spans ~2x2 texels in the HiZ.)
//
//   2. Sample the 4 corners of the AABB's UV rect from the HiZ.
//      The max of these samples is the farthest occluder depth.
//
//   3. Compare: if the block's CLOSEST NDC depth (minNdcDepth)
//      is GREATER than the max sampled HiZ depth, the block is
//      fully behind existing geometry → occluded.
//
// Temporal stability:
//   - The HiZ is built from the PREVIOUS frame's depth buffer.
//   - Newly revealed areas (camera motion) have far-plane depth
//     in the HiZ, so new blocks always pass (conservative).
//   - We use a small epsilon to avoid z-fighting at boundaries.
//   - Blocks that partially clip the near plane (minNdcDepth <= 0)
//     always pass (conservative).
//   - Blocks with no valid screen rect always pass.

__global__ void hizOcclusionCullKernel(
    const OocBlockDepthInfo* __restrict__ depthInfo,
    unsigned int                          count,
    cudaTextureObject_t                   hizTexture,
    int                                   hizWidth,
    int                                   hizHeight,
    unsigned int*            __restrict__ filteredBlockIds,
    unsigned int*            __restrict__ filteredCount)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    OocBlockDepthInfo blk = depthInfo[idx];

    // Conservative: always pass blocks with invalid screen rect or
    // blocks that intersect the near plane
    if (blk.screenMinU >= blk.screenMaxU ||
        blk.screenMinV >= blk.screenMaxV ||
        blk.minNdcDepth <= 0.0f)
    {
        unsigned int out = atomicAdd(filteredCount, 1u);
        filteredBlockIds[out] = blk.blockId;
        return;
    }

    // Compute the AABB size in HiZ texels
    float texelW = (blk.screenMaxU - blk.screenMinU) * static_cast<float>(hizWidth);
    float texelH = (blk.screenMaxV - blk.screenMinV) * static_cast<float>(hizHeight);
    float maxTexelSpan = fmaxf(texelW, texelH);

    // If the block covers too many texels, it's a large object — don't cull.
    // With a single-level HiZ, we can only efficiently test small-to-medium blocks.
    if (maxTexelSpan > static_cast<float>(hizWidth) * 0.5f)
    {
        unsigned int out = atomicAdd(filteredCount, 1u);
        filteredBlockIds[out] = blk.blockId;
        return;
    }

    // Sample the 4 corners of the AABB's rect in HiZ space.
    // tex2D with point filtering and unnormalized coords.
    float u0 = blk.screenMinU * static_cast<float>(hizWidth);
    float v0 = blk.screenMinV * static_cast<float>(hizHeight);
    float u1 = blk.screenMaxU * static_cast<float>(hizWidth);
    float v1 = blk.screenMaxV * static_cast<float>(hizHeight);

    // Clamp to valid range
    u0 = fmaxf(u0, 0.0f);
    v0 = fmaxf(v0, 0.0f);
    u1 = fminf(u1, static_cast<float>(hizWidth - 1));
    v1 = fminf(v1, static_cast<float>(hizHeight - 1));

    float d00 = tex2D<float>(hizTexture, u0, v0);
    float d10 = tex2D<float>(hizTexture, u1, v0);
    float d01 = tex2D<float>(hizTexture, u0, v1);
    float d11 = tex2D<float>(hizTexture, u1, v1);

    // Also sample center for better coverage
    float uc = (u0 + u1) * 0.5f;
    float vc = (v0 + v1) * 0.5f;
    float dcc = tex2D<float>(hizTexture, uc, vc);

    // Max depth from all samples = farthest occluder in this region
    float hizMaxDepth = fmaxf(fmaxf(fmaxf(d00, d10), fmaxf(d01, d11)), dcc);

    // The rasterizer writes NDC depth via (hit.z * proj22 + proj32) / (-hit.z).
    // However, the depth buffer is cleared with __float_as_uint(farPlane)
    // where farPlane is a large world-space distance, not NDC.
    // So cleared pixels have depth = farPlane (e.g. 1000.0), which is much
    // larger than any valid NDC depth (0..1).
    //
    // If hizMaxDepth is very large (> 1.0), this region has no rendered
    // geometry — the block should NOT be culled (conservative).
    if (hizMaxDepth > 1.0f)
    {
        unsigned int out = atomicAdd(filteredCount, 1u);
        filteredBlockIds[out] = blk.blockId;
        return;
    }

    // Compare: block's nearest point vs farthest occluder.
    // Small epsilon for z-fighting tolerance.
    const float HIZ_EPSILON = 0.0001f;
    if (blk.minNdcDepth > hizMaxDepth + HIZ_EPSILON)
        return;  // Fully occluded

    unsigned int out = atomicAdd(filteredCount, 1u);
    filteredBlockIds[out] = blk.blockId;
}

extern "C" void launchHizOcclusionCull(
    const OocBlockDepthInfo* d_depthInfo,
    unsigned int             count,
    cudaTextureObject_t      hizTexture,
    int                      hizWidth,
    int                      hizHeight,
    unsigned int*            d_filteredBlockIds,
    unsigned int*            d_filteredCount,
    cudaStream_t             stream)
{
    if (count == 0) return;
    dim3 block(256);
    dim3 grid((count + block.x - 1) / block.x);
    hizOcclusionCullKernel<<<grid, block, 0, stream>>>(
        d_depthInfo, count, hizTexture, hizWidth, hizHeight,
        d_filteredBlockIds, d_filteredCount);
}

// ═════════════════════════════════════════════════════════
// Kernel 3e: HiZ + Probabilistic combined occlusion culling
// ═════════════════════════════════════════════════════════
//
// Two-pass approach:
//   Pass 1 (parallel): HiZ test per-block → survivors written to
//          an intermediate buffer (d_hizPassIds) with their original
//          depth-sorted index preserved for stable ordering.
//   Pass 2 (single-thread): Iterate survivors front-to-back,
//          accumulating visibility via vc = (1 − Dc) * v_{c−1}
//          (Sharma et al. 2004, Eq. 2).  Blocks pass while v > threshold.

__global__ void hizProbabilisticPass1Kernel(
    const OocBlockDepthInfo* __restrict__ depthInfo,
    unsigned int                          count,
    cudaTextureObject_t                   hizTexture,
    int                                   hizWidth,
    int                                   hizHeight,
    unsigned int*            __restrict__ hizPassIndices,
    unsigned int*            __restrict__ hizPassCount)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    OocBlockDepthInfo blk = depthInfo[idx];

    if (blk.screenMinU >= blk.screenMaxU ||
        blk.screenMinV >= blk.screenMaxV ||
        blk.minNdcDepth <= 0.0f)
    {
        unsigned int out = atomicAdd(hizPassCount, 1u);
        hizPassIndices[out] = idx;
        return;
    }

    float texelW = (blk.screenMaxU - blk.screenMinU) * static_cast<float>(hizWidth);
    float texelH = (blk.screenMaxV - blk.screenMinV) * static_cast<float>(hizHeight);
    float maxTexelSpan = fmaxf(texelW, texelH);

    if (maxTexelSpan > static_cast<float>(hizWidth) * 0.5f)
    {
        unsigned int out = atomicAdd(hizPassCount, 1u);
        hizPassIndices[out] = idx;
        return;
    }

    float u0 = fmaxf(blk.screenMinU * static_cast<float>(hizWidth), 0.0f);
    float v0 = fmaxf(blk.screenMinV * static_cast<float>(hizHeight), 0.0f);
    float u1 = fminf(blk.screenMaxU * static_cast<float>(hizWidth),
                      static_cast<float>(hizWidth - 1));
    float v1 = fminf(blk.screenMaxV * static_cast<float>(hizHeight),
                      static_cast<float>(hizHeight - 1));

    float d00 = tex2D<float>(hizTexture, u0, v0);
    float d10 = tex2D<float>(hizTexture, u1, v0);
    float d01 = tex2D<float>(hizTexture, u0, v1);
    float d11 = tex2D<float>(hizTexture, u1, v1);
    float uc = (u0 + u1) * 0.5f;
    float vc = (v0 + v1) * 0.5f;
    float dcc = tex2D<float>(hizTexture, uc, vc);

    float hizMaxDepth = fmaxf(fmaxf(fmaxf(d00, d10), fmaxf(d01, d11)), dcc);

    if (hizMaxDepth > 1.0f)
    {
        unsigned int out = atomicAdd(hizPassCount, 1u);
        hizPassIndices[out] = idx;
        return;
    }

    const float HIZ_EPSILON = 0.0001f;
    if (blk.minNdcDepth > hizMaxDepth + HIZ_EPSILON)
        return;

    unsigned int out = atomicAdd(hizPassCount, 1u);
    hizPassIndices[out] = idx;
}

static constexpr int PROB_TILE_GRID = 16;
static constexpr int PROB_NUM_TILES = PROB_TILE_GRID * PROB_TILE_GRID;

__global__ void hizProbabilisticPass2TiledKernel(
    const OocBlockDepthInfo* __restrict__ depthInfo,
    const unsigned int*      __restrict__ hizPassIndices,
    unsigned int                          hizPassCount,
    unsigned int*            __restrict__ filteredBlockIds,
    unsigned int*            __restrict__ filteredCount)
{
    int tid   = threadIdx.x;
    int tileX = tid % PROB_TILE_GRID;
    int tileY = tid / PROB_TILE_GRID;

    float screenW = static_cast<float>(oocCst.screenWidth);
    float screenH = static_cast<float>(oocCst.screenHeight);
    float tileW   = screenW / static_cast<float>(PROB_TILE_GRID);
    float tileH   = screenH / static_cast<float>(PROB_TILE_GRID);
    float invTileArea = 1.0f / (tileW * tileH);

    float myMinX = static_cast<float>(tileX) * tileW;
    float myMinY = static_cast<float>(tileY) * tileH;
    float myMaxX = myMinX + tileW;
    float myMaxY = myMinY + tileH;

    float v = 1.0f;

    float capExponent = -logf(fmaxf(oocCst.visibilityThreshold, 1e-6f))
                        / (2.0f * cbrtf(fmaxf(static_cast<float>(oocCst.atomsPerBlock), 1.0f)));

    __shared__ int s_visible;

    for (unsigned int c = 0; c < hizPassCount; c++)
    {
        unsigned int origIdx = hizPassIndices[c];
        OocBlockDepthInfo blk = depthInfo[origIdx];

        float blockW = blk.screenMaxX - blk.screenMinX;
        float blockH = blk.screenMaxY - blk.screenMinY;
        bool degenerate = (blockW <= 0.0f || blockH <= 0.0f);

        bool overlaps = !degenerate &&
                        (blk.screenMaxX > myMinX && blk.screenMinX < myMaxX &&
                         blk.screenMaxY > myMinY && blk.screenMinY < myMaxY);

        bool iPass = degenerate || (overlaps && v > oocCst.visibilityThreshold);

        if (tid == 0) s_visible = 0;
        __syncthreads();

        if (iPass) atomicMax(&s_visible, 1);
        __syncthreads();

        if (s_visible && tid == 0)
        {
            unsigned int out = atomicAdd(filteredCount, 1u);
            filteredBlockIds[out] = blk.blockId;
        }

        if (overlaps)
        {
            float ox = fminf(blk.screenMaxX, myMaxX) - fmaxf(blk.screenMinX, myMinX);
            float oy = fminf(blk.screenMaxY, myMaxY) - fmaxf(blk.screenMinY, myMinY);
            float fraction = (ox * oy) / (blockW * blockH);
            float Dc_raw = blk.projectedArea * fraction * invTileArea;
            v *= expf(-fminf(Dc_raw, capExponent));
        }
    }
}

extern "C" void launchHizProbabilisticOcclusion(
    const OocBlockDepthInfo* d_depthInfo,
    unsigned int             count,
    cudaTextureObject_t      hizTexture,
    int                      hizWidth,
    int                      hizHeight,
    unsigned int*            d_hizPassIndices,
    unsigned int*            d_hizPassCount,
    unsigned int*            d_filteredBlockIds,
    unsigned int*            d_filteredCount,
    cudaStream_t             stream)
{
    if (count == 0) return;

    cudaMemsetAsync(d_hizPassCount, 0, sizeof(unsigned int), stream);

    dim3 block(256);
    dim3 grid((count + block.x - 1) / block.x);
    hizProbabilisticPass1Kernel<<<grid, block, 0, stream>>>(
        d_depthInfo, count, hizTexture, hizWidth, hizHeight,
        d_hizPassIndices, d_hizPassCount);

    unsigned int hizSurvivors = 0;
    cudaMemcpyAsync(&hizSurvivors, d_hizPassCount, sizeof(unsigned int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (hizSurvivors == 0) return;

    thrust::device_ptr<unsigned int> idxPtr(d_hizPassIndices);
    thrust::sort(thrust::cuda::par.on(stream),
                 idxPtr, idxPtr + hizSurvivors);

    hizProbabilisticPass2TiledKernel<<<1, PROB_NUM_TILES, 0, stream>>>(
        d_depthInfo, d_hizPassIndices, hizSurvivors,
        d_filteredBlockIds, d_filteredCount);
}

// ═════════════════════════════════════════════════════════
// Kernel 3d: No occlusion (pass-through)
// ═════════════════════════════════════════════════════════

__global__ void noOcclusionPassthroughKernel(
    const OocBlockDepthInfo* __restrict__ depthInfo,
    unsigned int                          count,
    unsigned int*            __restrict__ filteredBlockIds,
    unsigned int*            __restrict__ filteredCount)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    unsigned int out = atomicAdd(filteredCount, 1u);
    filteredBlockIds[out] = depthInfo[idx].blockId;
}

extern "C" void launchNoOcclusionPassthrough(
    const OocBlockDepthInfo* d_depthInfo,
    unsigned int             count,
    unsigned int*            d_filteredBlockIds,
    unsigned int*            d_filteredCount,
    cudaStream_t             stream)
{
    if (count == 0) return;
    dim3 block(256);
    dim3 grid((count + block.x - 1) / block.x);
    noOcclusionPassthroughKernel<<<grid, block, 0, stream>>>(
        d_depthInfo, count, d_filteredBlockIds, d_filteredCount);
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

    const glm::vec4* src = atomPool + static_cast<size_t>(slot) * oocCst.atomsPerBlock;
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

// ═════════════════════════════════════════════════════════
// HiZ downsample kernel (builds HiZ from full-res depth buffer)
// ═════════════════════════════════════════════════════════

__global__ void hizDownsampleKernel(
    const unsigned int* __restrict__ depthBuffer,
    cudaSurfaceObject_t              hizSurface,
    int                              fullWidth,
    int                              fullHeight,
    int                              hizWidth,
    int                              hizHeight)
{
    int hx = blockIdx.x * blockDim.x + threadIdx.x;
    int hy = blockIdx.y * blockDim.y + threadIdx.y;
    if (hx >= hizWidth || hy >= hizHeight) return;

    int scaleX = (fullWidth  + hizWidth  - 1) / hizWidth;
    int scaleY = (fullHeight + hizHeight - 1) / hizHeight;

    int x0 = hx * scaleX;
    int y0 = hy * scaleY;
    int x1 = min(x0 + scaleX, fullWidth);
    int y1 = min(y0 + scaleY, fullHeight);

    float maxD = 0.0f;
    for (int y = y0; y < y1; y++) {
        for (int x = x0; x < x1; x++) {
            float d = __uint_as_float(depthBuffer[y * fullWidth + x]);
            maxD = fmaxf(maxD, d);
        }
    }

    surf2Dwrite(maxD, hizSurface, hx * static_cast<int>(sizeof(float)), hy);
}

extern "C" void launchHizDownsample(
    const unsigned int*  d_depthBuffer,
    cudaSurfaceObject_t  hizSurface,
    int fullWidth, int fullHeight,
    int hizWidth, int hizHeight,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((hizWidth + 15) / 16, (hizHeight + 15) / 16);
    hizDownsampleKernel<<<grid, block, 0, stream>>>(
        d_depthBuffer, hizSurface, fullWidth, fullHeight, hizWidth, hizHeight);
}
