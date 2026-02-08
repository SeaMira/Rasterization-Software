/**
 * @file tiled_raster_binning.cu
 * @brief Tiled rasterization using tileOffsets + sorted entity indices (no billboards).
 * Loads sphere/cylinder data from original buffers; recomputes bbox in shared memory.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>
#include "hybrid_binning_types.h"
#include "geometry/cylinder/cylinder.h"

extern __constant__ HybridConstants hybridCst;

#define TILE_SIZE 16
#define SHARED_BATCH 64
#define HEAVY_TILE_THRESHOLD 256

/** Work-stealing counters for persistent-thread bin-split kernels.
 *  g_nextLightWorkItem: index into the compact light-tile list.
 *  g_nextHeavyWorkItem: index into the expanded heavy work-item list.
 *  Must be reset to 0 before each kernel launch. */
__device__ unsigned int g_nextLightWorkItem;
__device__ unsigned int g_nextHeavyWorkItem;

__device__ inline float iSphereTiled(const glm::vec3& ro, const glm::vec3& rd,
                                     const glm::vec3& center, float radius) {
    glm::vec3 oc = ro - center;
    float b = glm::dot(oc, rd);
    float c = glm::dot(oc, oc) - radius * radius;
    float h = b * b - c;
    if (h < 0.0f) return -1.0f;
    return -b - sqrtf(h);
}

__device__ inline glm::vec3 computeRayDirTiled(int px, int py, float fovTan, float halfFovTan) {
    glm::vec2 p = (-glm::vec2(hybridCst.screenWidth, hybridCst.screenHeight) +
                   2.0f * glm::vec2(px, py)) / glm::vec2(hybridCst.screenWidth, hybridCst.screenHeight);
    return glm::normalize(p.x * hybridCst.right * halfFovTan +
                          p.y * hybridCst.up * fovTan + hybridCst.front);
}

__device__ inline void computeSphereBBoxTiled(const glm::vec4& spherePosR,
    int& bboxMinX, int& bboxMinY, int& bboxMaxX, int& bboxMaxY) 
{
    glm::vec3 spherePos = glm::vec3(spherePosR);
    float radius = spherePosR.w;
    glm::vec4 camSpace4 = hybridCst.view * glm::vec4(spherePos, 1.0f);
    glm::vec3 cameraSpaceSphere = glm::vec3(camSpace4);
    glm::vec3 normCamSpaceSphere = glm::normalize(cameraSpaceSphere);
    glm::vec3 camImposPos = cameraSpaceSphere - normCamSpaceSphere * radius;

    float dist = glm::length(cameraSpaceSphere) + 1e-6f;
    float sinAngle = __fdividef(radius, dist);
    float tanAngle = tanf(asinf(fminf(sinAngle, 0.999f)));
    float quadScale = tanAngle * glm::length(camImposPos);

    glm::vec3 upVec(0.0f, 1.0f, 0.0f);
    glm::vec3 impU = glm::normalize(glm::cross(normCamSpaceSphere, upVec));
    if (glm::length(impU) < 0.001f) impU = glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 impV = glm::cross(impU, normCamSpaceSphere) * quadScale;
    impU *= quadScale;

    glm::vec3 corners[4] = {
        camImposPos + impU + impV, camImposPos - impU + impV,
        camImposPos + impU - impV, camImposPos - impU - impV
    };

    glm::vec2 minC(1e6f), maxC(-1e6f);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        glm::vec4 clip = hybridCst.proj * glm::vec4(corners[i], 1.0f);
        float iw = __fdividef(1.0f, clip.w);
        float x = clip.x * iw;
        float y = clip.y * iw;
        minC.x = fminf(minC.x, x); minC.y = fminf(minC.y, y);
        maxC.x = fmaxf(maxC.x, x); maxC.y = fmaxf(maxC.y, y);
    }

    bboxMinX = __float2int_rd(fmaf(minC.x, 0.5f, 0.5f) * hybridCst.screenWidth);
    bboxMinY = __float2int_rd(fmaf(minC.y, 0.5f, 0.5f) * hybridCst.screenHeight);
    bboxMaxX = __float2int_ru(fmaf(maxC.x, 0.5f, 0.5f) * hybridCst.screenWidth);
    bboxMaxY = __float2int_ru(fmaf(maxC.y, 0.5f, 0.5f) * hybridCst.screenHeight);
}

/** Classify tiles into light (single-block) and heavy (multi-block) lists.
 *  - Empty tiles: skipped.
 *  - Light tiles (entityCount in (0, HEAVY_TILE_THRESHOLD]): 1 entry in d_lightTileList.
 *  - Heavy tiles (entityCount > HEAVY_TILE_THRESHOLD): ceil(entityCount/SHARED_BATCH)
 *    work-items in d_heavyTileIndices + d_heavyBatchStarts.
 *  d_tileClassifyCounts[0] = total light tiles, [1] = total heavy work items. */
__global__ void classifyTilesKernel(
    const unsigned int* __restrict__ d_tile_offsets,
    unsigned int* __restrict__ d_lightTileList,
    unsigned int* __restrict__ d_heavyTileIndices,
    unsigned int* __restrict__ d_heavyBatchStarts,
    unsigned int* __restrict__ d_tileClassifyCounts,
    unsigned int totalTiles)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= totalTiles) return;
    unsigned int entityCount = d_tile_offsets[tid + 1] - d_tile_offsets[tid];
    if (entityCount == 0) return;

    if (entityCount <= HEAVY_TILE_THRESHOLD) {
        unsigned int idx = atomicAdd(&d_tileClassifyCounts[0], 1u);
        d_lightTileList[idx] = tid;
    } else {
        unsigned int numBatches = (entityCount + SHARED_BATCH - 1) / SHARED_BATCH;
        unsigned int base = atomicAdd(&d_tileClassifyCounts[1], numBatches);
        for (unsigned int b = 0; b < numBatches; b++) {
            d_heavyTileIndices[base + b] = tid;
            d_heavyBatchStarts[base + b] = b * SHARED_BATCH;
        }
    }
}

/** Light sphere kernel: persistent threads, 1 tile per work-steal.
 *  Only ONE block ever processes a given tile, so no atomicMin needed. */
__global__ void lightSphereRasterKernel(
    const glm::vec4* __restrict__ d_spheres,
    const unsigned int* __restrict__ d_tile_offsets,
    const unsigned long long* __restrict__ d_tile_entity_pairs_sorted,
    const unsigned int* __restrict__ d_lightTileList,
    const unsigned int* __restrict__ d_tileClassifyCounts,
    unsigned int* __restrict__ depthBuffer,
    cudaSurfaceObject_t outputImage)
{
    const float aspectRatio = (float)hybridCst.screenWidth / (float)hybridCst.screenHeight;
    const float fovRad = glm::radians(hybridCst.fov);
    const float fovTan = tanf(fovRad * 0.5f);
    const float halfFovTan = fovTan * aspectRatio;
    const float proj22 = hybridCst.proj[2][2];
    const float proj32 = hybridCst.proj[3][2];

    __shared__ unsigned int s_workIdx;
    __shared__ glm::vec4 shPosR[SHARED_BATCH];

    const unsigned int totalLightTiles = d_tileClassifyCounts[0];

    while (true) {
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            s_workIdx = atomicAdd(&g_nextLightWorkItem, 1u);
        }
        __syncthreads();

        if (s_workIdx >= totalLightTiles) return;

        const unsigned int tileIdx = d_lightTileList[s_workIdx];
        const unsigned int entityStart = d_tile_offsets[tileIdx];
        const unsigned int entityEnd   = d_tile_offsets[tileIdx + 1];
        const unsigned int entityCount = entityEnd - entityStart;

        const unsigned int tileX = tileIdx % hybridCst.tilesX;
        const unsigned int tileY = tileIdx / hybridCst.tilesX;
        const int pixelX = tileX * TILE_SIZE + threadIdx.x;
        const int pixelY = tileY * TILE_SIZE + threadIdx.y;
        const bool validPixel = (pixelX < hybridCst.screenWidth && pixelY < hybridCst.screenHeight);

        glm::vec3 rd, ray;
        float minDepth = 1e30f;
        if (validPixel) {
            rd  = computeRayDirTiled(pixelX, pixelY, fovTan, halfFovTan);
            ray = hybridCst.rayStart + (float)pixelX * hybridCst.dx + (float)pixelY * hybridCst.dy;
            minDepth = __uint_as_float(depthBuffer[pixelY * hybridCst.screenWidth + pixelX]);
        }
        glm::vec3 finalColor(0.0f);
        bool hasHit = false;

        // Process ALL batches for this light tile within this single block
        const unsigned int numBatches = (entityCount + SHARED_BATCH - 1) / SHARED_BATCH;
        for (unsigned int batch = 0; batch < numBatches; batch++) {
            unsigned int batchStart = batch * SHARED_BATCH;
            unsigned int localIdx = threadIdx.y * TILE_SIZE + threadIdx.x;
            if (localIdx < SHARED_BATCH) {
                unsigned int entityIdx = batchStart + localIdx;
                if (entityIdx < entityCount) {
                    unsigned long long pair = d_tile_entity_pairs_sorted[entityStart + entityIdx];
                    unsigned int sphereIndex = (unsigned int)(pair & 0xFFFFFFFFu);
                    shPosR[localIdx] = d_spheres[sphereIndex];
                }
            }
            __syncthreads();

            if (validPixel) {
                unsigned int batchCount = min(SHARED_BATCH, entityCount - batchStart);
                for (unsigned int i = 0; i < batchCount; i++) {
                    glm::vec4 posR = shPosR[i];
                    float t = iSphereTiled(hybridCst.cameraPos, rd, glm::vec3(posR), posR.w);
                    if (t > 0.0f) {
                        glm::vec3 hit = ray * t;
                        float depth = __fdividef(fmaf(hit.z, proj22, proj32), -hit.z);
                        if (depth < minDepth) {
                            minDepth = depth;
                            hasHit = true;
                            glm::vec3 worldHit = hybridCst.cameraPos + rd * t;
                            glm::vec3 normal = glm::normalize(worldHit - glm::vec3(posR));
                            float lambert = fmaxf(0.0f, glm::dot(normal, -glm::normalize(rd * t)));
                            finalColor = glm::vec3(atomsColor[0], atomsColor[1], atomsColor[2]) * lambert * diffuse;
                        }
                    }
                }
            }
            __syncthreads();
        }

        // Single block owns this tile — direct write, no atomicMin
        if (validPixel && hasHit) {
            int pixelIdx = pixelY * hybridCst.screenWidth + pixelX;
            unsigned int depthU = __float_as_uint(minDepth);
            depthBuffer[pixelIdx] = depthU;
            uchar4 pixel = make_uchar4(
                (unsigned char)(finalColor.x * 255.0f),
                (unsigned char)(finalColor.y * 255.0f),
                (unsigned char)(finalColor.z * 255.0f), 255);
            surf2Dwrite(pixel, outputImage, pixelX * sizeof(uchar4), pixelY);
        }
        __syncthreads();
    }
}

/** Heavy sphere kernel: persistent threads, 1 batch per work-steal.
 *  Multiple blocks may process the same tile -> atomicMin on depth buffer. */
__global__ void heavySphereRasterKernel(
    const glm::vec4* __restrict__ d_spheres,
    const unsigned int* __restrict__ d_tile_offsets,
    const unsigned long long* __restrict__ d_tile_entity_pairs_sorted,
    const unsigned int* __restrict__ d_heavyTileIndices,
    const unsigned int* __restrict__ d_heavyBatchStarts,
    const unsigned int* __restrict__ d_tileClassifyCounts,
    unsigned int* __restrict__ depthBuffer,
    cudaSurfaceObject_t outputImage)
{
    const float aspectRatio = (float)hybridCst.screenWidth / (float)hybridCst.screenHeight;
    const float fovRad = glm::radians(hybridCst.fov);
    const float fovTan = tanf(fovRad * 0.5f);
    const float halfFovTan = fovTan * aspectRatio;
    const float proj22 = hybridCst.proj[2][2];
    const float proj32 = hybridCst.proj[3][2];

    __shared__ unsigned int s_workIdx;
    __shared__ unsigned int s_tileIdx;
    __shared__ unsigned int s_batchStart;
    __shared__ unsigned int s_batchCount;
    __shared__ glm::vec4 shPosR[SHARED_BATCH];

    const unsigned int totalHeavyWorkItems = d_tileClassifyCounts[1];

    while (true) {
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            s_workIdx = atomicAdd(&g_nextHeavyWorkItem, 1u);
        }
        __syncthreads();

        if (s_workIdx >= totalHeavyWorkItems) return;

        // Decode which tile and which batch this work item represents
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            unsigned int tileIdx = d_heavyTileIndices[s_workIdx];
            unsigned int bStart  = d_heavyBatchStarts[s_workIdx];
            unsigned int entityCount = d_tile_offsets[tileIdx + 1] - d_tile_offsets[tileIdx];
            s_tileIdx    = tileIdx;
            s_batchStart = bStart;
            s_batchCount = min((unsigned int)SHARED_BATCH, entityCount - min(bStart, entityCount));
        }
        __syncthreads();

        const unsigned int tileIdx    = s_tileIdx;
        const unsigned int batchStart = s_batchStart;
        const unsigned int batchCount = s_batchCount;

        if (batchCount == 0) { __syncthreads(); continue; }

        const unsigned int tileX = tileIdx % hybridCst.tilesX;
        const unsigned int tileY = tileIdx / hybridCst.tilesX;
        const int pixelX = tileX * TILE_SIZE + threadIdx.x;
        const int pixelY = tileY * TILE_SIZE + threadIdx.y;
        const bool validPixel = (pixelX < hybridCst.screenWidth && pixelY < hybridCst.screenHeight);

        // Load one batch of entities into shared memory
        const unsigned int entityStart = d_tile_offsets[tileIdx];
        const unsigned int entityCount = d_tile_offsets[tileIdx + 1] - entityStart;
        const unsigned int localIdx = threadIdx.y * TILE_SIZE + threadIdx.x;
        if (localIdx < SHARED_BATCH) {
            unsigned int entityIdx = batchStart + localIdx;
            if (entityIdx < entityCount) {
                unsigned long long pair = d_tile_entity_pairs_sorted[entityStart + entityIdx];
                unsigned int sphereIndex = (unsigned int)(pair & 0xFFFFFFFFu);
                shPosR[localIdx] = d_spheres[sphereIndex];
            }
        }
        __syncthreads();

        // Ray-trace this single batch; atomicMin for depth since multiple blocks share the tile
        if (validPixel) {
            glm::vec3 rd  = computeRayDirTiled(pixelX, pixelY, fovTan, halfFovTan);
            glm::vec3 ray = hybridCst.rayStart + (float)pixelX * hybridCst.dx + (float)pixelY * hybridCst.dy;
            const int pixelIdx = pixelY * hybridCst.screenWidth + pixelX;

            for (unsigned int i = 0; i < batchCount; i++) {
                glm::vec4 posR = shPosR[i];
                float t = iSphereTiled(hybridCst.cameraPos, rd, glm::vec3(posR), posR.w);
                if (t > 0.0f) {
                    glm::vec3 hit = ray * t;
                    float depth = __fdividef(fmaf(hit.z, proj22, proj32), -hit.z);
                    unsigned int depthU = __float_as_uint(depth);
                    unsigned int old = atomicMin(&depthBuffer[pixelIdx], depthU);
                    if (depthU < old) {
                        glm::vec3 worldHit = hybridCst.cameraPos + rd * t;
                        glm::vec3 normal = glm::normalize(worldHit - glm::vec3(posR));
                        float lambert = fmaxf(0.0f, glm::dot(normal, -glm::normalize(rd * t)));
                        glm::vec3 color = glm::vec3(atomsColor[0], atomsColor[1], atomsColor[2]) * lambert * diffuse;
                        uchar4 pixel = make_uchar4(
                            (unsigned char)(color.x * 255.0f),
                            (unsigned char)(color.y * 255.0f),
                            (unsigned char)(color.z * 255.0f), 255);
                        surf2Dwrite(pixel, outputImage, pixelX * sizeof(uchar4), pixelY);
                    }
                }
            }
        }
        __syncthreads();
    }
}



// Cylinder: ray-cylinder and bbox from quad (simplified - reuse logic from hybrid tiled)
__device__ inline glm::vec4 iCylinderTiled(const glm::vec3& ro, const glm::vec3& rd,
    const glm::vec3& pa, const glm::vec3& pb, float ra) 
    {
    glm::vec3 ba = pb - pa, oc = ro - pa;
    float baba = glm::dot(ba, ba), bard = glm::dot(ba, rd), baoc = glm::dot(ba, oc);
    
    float k2 = fmaf(bard, -bard, baba);
    float k1 = fmaf(glm::dot(oc,rd), baba, -baoc*bard);
    float k0 = fmaf(baba, glm::dot(oc,oc), fmaf(- ra*ra, baba, -baoc*baoc));
    
    float h = fmaf(k1, k1, -k2*k0);
    if (h < 0.0f) return glm::vec4(-1.0f);
    h = sqrtf(h);
    float t = (-k1 - h) / k2;
    float y = fmaf(t, bard, baoc);
    if (y > 0.0f && y < baba) return glm::vec4(t, oc + t * rd - ba * y / baba);
    return glm::vec4(-1.0f);
}

/** Light cylinder kernel: persistent threads, 1 tile per work-steal. No atomicMin. */
__global__ void lightCylinderRasterKernel(
    const Cylinder* __restrict__ d_cylinders,
    const unsigned int* __restrict__ d_tile_offsets,
    const unsigned long long* __restrict__ d_tile_entity_pairs_sorted,
    const unsigned int* __restrict__ d_lightTileList,
    const unsigned int* __restrict__ d_tileClassifyCounts,
    unsigned int* __restrict__ depthBuffer,
    cudaSurfaceObject_t outputImage)
{
    const float aspectRatio = (float)hybridCst.screenWidth / (float)hybridCst.screenHeight;
    const float fovRad = glm::radians(hybridCst.fov);
    const float fovTan = tanf(fovRad * 0.5f);
    const float halfFovTan = fovTan * aspectRatio;
    const float proj22 = hybridCst.proj[2][2];
    const float proj32 = hybridCst.proj[3][2];

    __shared__ unsigned int s_workIdx;
    __shared__ glm::vec4 shPa[SHARED_BATCH], shPb[SHARED_BATCH];

    const unsigned int totalLightTiles = d_tileClassifyCounts[0];

    while (true) {
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            s_workIdx = atomicAdd(&g_nextLightWorkItem, 1u);
        }
        __syncthreads();

        if (s_workIdx >= totalLightTiles) return;

        const unsigned int tileIdx = d_lightTileList[s_workIdx];
        const unsigned int entityStart = d_tile_offsets[tileIdx];
        const unsigned int entityEnd   = d_tile_offsets[tileIdx + 1];
        const unsigned int entityCount = entityEnd - entityStart;

        const unsigned int tileX = tileIdx % hybridCst.tilesX;
        const unsigned int tileY = tileIdx / hybridCst.tilesX;
        const int pixelX = tileX * TILE_SIZE + threadIdx.x;
        const int pixelY = tileY * TILE_SIZE + threadIdx.y;
        const bool validPixel = (pixelX < hybridCst.screenWidth && pixelY < hybridCst.screenHeight);

        glm::vec3 rd, ray;
        float minDepth = 1e30f;
        if (validPixel) {
            rd  = computeRayDirTiled(pixelX, pixelY, fovTan, halfFovTan);
            ray = hybridCst.rayStart + (float)pixelX * hybridCst.dx + (float)pixelY * hybridCst.dy;
            minDepth = __uint_as_float(depthBuffer[pixelY * hybridCst.screenWidth + pixelX]);
        }
        glm::vec3 finalColor(0.0f);
        bool hasHit = false;

        const unsigned int numBatches = (entityCount + SHARED_BATCH - 1) / SHARED_BATCH;
        for (unsigned int batch = 0; batch < numBatches; batch++) {
            unsigned int batchStart = batch * SHARED_BATCH;
            unsigned int localIdx = threadIdx.y * TILE_SIZE + threadIdx.x;
            if (localIdx < SHARED_BATCH) {
                unsigned int entityIdx = batchStart + localIdx;
                if (entityIdx < entityCount) {
                    unsigned long long pair = d_tile_entity_pairs_sorted[entityStart + entityIdx];
                    unsigned int cylIndex = (unsigned int)(pair & 0xFFFFFFFFu);
                    Cylinder cyl = d_cylinders[cylIndex];
                    shPa[localIdx] = cyl.pa_r;
                    shPb[localIdx] = cyl.pb_r;
                }
            }
            __syncthreads();

            if (validPixel) {
                unsigned int batchCount = min(SHARED_BATCH, entityCount - batchStart);
                for (unsigned int i = 0; i < batchCount; i++) {
                    glm::vec3 pa = glm::vec3(shPa[i]), pb = glm::vec3(shPb[i]);
                    float ra = shPa[i].w;
                    glm::vec4 tnor = iCylinderTiled(hybridCst.cameraPos, rd, pa, pb, ra);
                    if (tnor.x > 0.0f) {
                        float t = tnor.x;
                        glm::vec3 hit = ray * t;
                        float depth = __fdividef(fmaf(hit.z, proj22, proj32), -hit.z);
                        if (depth < minDepth) {
                            minDepth = depth;
                            hasHit = true;
                            glm::vec3 normal = glm::normalize(glm::vec3(tnor.y, tnor.z, tnor.w));
                            float lambert = glm::max(0.0f, glm::dot(normal, -glm::normalize(rd * t)));
                            finalColor = glm::vec3(bondsColor[0], bondsColor[1], bondsColor[2]) * lambert * diffuse;
                        }
                    }
                }
            }
            __syncthreads();
        }

        if (validPixel && hasHit) {
            int pixelIdx = pixelY * hybridCst.screenWidth + pixelX;
            unsigned int depthU = __float_as_uint(minDepth);
            depthBuffer[pixelIdx] = depthU;
            uchar4 pixel = make_uchar4(
                (unsigned char)(finalColor.x * 255.0f),
                (unsigned char)(finalColor.y * 255.0f),
                (unsigned char)(finalColor.z * 255.0f), 255);
            surf2Dwrite(pixel, outputImage, pixelX * sizeof(uchar4), pixelY);
        }
        __syncthreads();
    }
}

/** Heavy cylinder kernel: persistent threads, 1 batch per work-steal. atomicMin depth. */
__global__ void heavyCylinderRasterKernel(
    const Cylinder* __restrict__ d_cylinders,
    const unsigned int* __restrict__ d_tile_offsets,
    const unsigned long long* __restrict__ d_tile_entity_pairs_sorted,
    const unsigned int* __restrict__ d_heavyTileIndices,
    const unsigned int* __restrict__ d_heavyBatchStarts,
    const unsigned int* __restrict__ d_tileClassifyCounts,
    unsigned int* __restrict__ depthBuffer,
    cudaSurfaceObject_t outputImage)
{
    const float aspectRatio = (float)hybridCst.screenWidth / (float)hybridCst.screenHeight;
    const float fovRad = glm::radians(hybridCst.fov);
    const float fovTan = tanf(fovRad * 0.5f);
    const float halfFovTan = fovTan * aspectRatio;
    const float proj22 = hybridCst.proj[2][2];
    const float proj32 = hybridCst.proj[3][2];

    __shared__ unsigned int s_workIdx;
    __shared__ unsigned int s_tileIdx;
    __shared__ unsigned int s_batchStart;
    __shared__ unsigned int s_batchCount;
    __shared__ glm::vec4 shPa[SHARED_BATCH], shPb[SHARED_BATCH];

    const unsigned int totalHeavyWorkItems = d_tileClassifyCounts[1];

    while (true) {
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            s_workIdx = atomicAdd(&g_nextHeavyWorkItem, 1u);
        }
        __syncthreads();

        if (s_workIdx >= totalHeavyWorkItems) return;

        if (threadIdx.x == 0 && threadIdx.y == 0) {
            unsigned int tileIdx = d_heavyTileIndices[s_workIdx];
            unsigned int bStart  = d_heavyBatchStarts[s_workIdx];
            unsigned int entityCount = d_tile_offsets[tileIdx + 1] - d_tile_offsets[tileIdx];
            s_tileIdx    = tileIdx;
            s_batchStart = bStart;
            s_batchCount = min((unsigned int)SHARED_BATCH, entityCount - min(bStart, entityCount));
        }
        __syncthreads();

        const unsigned int tileIdx    = s_tileIdx;
        const unsigned int batchStart = s_batchStart;
        const unsigned int batchCount = s_batchCount;

        if (batchCount == 0) { __syncthreads(); continue; }

        const unsigned int tileX = tileIdx % hybridCst.tilesX;
        const unsigned int tileY = tileIdx / hybridCst.tilesX;
        const int pixelX = tileX * TILE_SIZE + threadIdx.x;
        const int pixelY = tileY * TILE_SIZE + threadIdx.y;
        const bool validPixel = (pixelX < hybridCst.screenWidth && pixelY < hybridCst.screenHeight);

        const unsigned int entityStart = d_tile_offsets[tileIdx];
        const unsigned int entityCount = d_tile_offsets[tileIdx + 1] - entityStart;
        const unsigned int localIdx = threadIdx.y * TILE_SIZE + threadIdx.x;
        if (localIdx < SHARED_BATCH) {
            unsigned int entityIdx = batchStart + localIdx;
            if (entityIdx < entityCount) {
                unsigned long long pair = d_tile_entity_pairs_sorted[entityStart + entityIdx];
                unsigned int cylIndex = (unsigned int)(pair & 0xFFFFFFFFu);
                Cylinder cyl = d_cylinders[cylIndex];
                shPa[localIdx] = cyl.pa_r;
                shPb[localIdx] = cyl.pb_r;
            }
        }
        __syncthreads();

        if (validPixel) {
            glm::vec3 rd  = computeRayDirTiled(pixelX, pixelY, fovTan, halfFovTan);
            glm::vec3 ray = hybridCst.rayStart + (float)pixelX * hybridCst.dx + (float)pixelY * hybridCst.dy;
            const int pixelIdx = pixelY * hybridCst.screenWidth + pixelX;

            for (unsigned int i = 0; i < batchCount; i++) {
                glm::vec3 pa = glm::vec3(shPa[i]), pb = glm::vec3(shPb[i]);
                float ra = shPa[i].w;
                glm::vec4 tnor = iCylinderTiled(hybridCst.cameraPos, rd, pa, pb, ra);
                if (tnor.x > 0.0f) {
                    float t = tnor.x;
                    glm::vec3 hit = ray * t;
                    float depth = __fdividef(fmaf(hit.z, proj22, proj32), -hit.z);
                    unsigned int depthU = __float_as_uint(depth);
                    unsigned int old = atomicMin(&depthBuffer[pixelIdx], depthU);
                    if (depthU < old) {
                        glm::vec3 normal = glm::normalize(glm::vec3(tnor.y, tnor.z, tnor.w));
                        float lambert = glm::max(0.0f, glm::dot(normal, -glm::normalize(rd * t)));
                        glm::vec3 color = glm::vec3(bondsColor[0], bondsColor[1], bondsColor[2]) * lambert * diffuse;
                        uchar4 pixel = make_uchar4(
                            (unsigned char)(color.x * 255.0f),
                            (unsigned char)(color.y * 255.0f),
                            (unsigned char)(color.z * 255.0f), 255);
                        surf2Dwrite(pixel, outputImage, pixelX * sizeof(uchar4), pixelY);
                    }
                }
            }
        }
        __syncthreads();
    }
}

// Helper: query SM count (cached)
static int getNumSMs() {
    static int numSMs = 0;
    if (numSMs == 0) {
        int device = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device);
    }
    return numSMs;
}

extern "C" void launchTiledSphereRasterBinning(
    const glm::vec4* d_spheres,
    const unsigned int* d_tile_offsets,
    const unsigned long long* d_tile_entity_pairs_sorted,
    unsigned int* d_depthBuffer,
    cudaSurfaceObject_t outputImage,
    unsigned int tilesX,
    unsigned int tilesY,
    unsigned int* d_lightTileList,
    unsigned int* d_heavyTileIndices,
    unsigned int* d_heavyBatchStarts,
    unsigned int* d_tileClassifyCounts,
    cudaStream_t stream)
{
    unsigned int totalTiles = tilesX * tilesY;

    // Classify tiles into light and heavy lists
    cudaMemsetAsync(d_tileClassifyCounts, 0, 2 * sizeof(unsigned int), stream);
    {
        int threads = 256;
        int blocks = (totalTiles + threads - 1) / threads;
        classifyTilesKernel<<<blocks, threads, 0, stream>>>(
            d_tile_offsets, d_lightTileList, d_heavyTileIndices,
            d_heavyBatchStarts, d_tileClassifyCounts, totalTiles);
    }

    int numSMs = getNumSMs();
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(numSMs * 2, 1);

    // Light tiles: persistent threads, 1 tile per work-steal (no atomicMin)
    {
        unsigned int zero = 0;
        cudaMemcpyToSymbolAsync(g_nextLightWorkItem, &zero, sizeof(unsigned int), 0, cudaMemcpyHostToDevice, stream);
        lightSphereRasterKernel<<<grid, block, 0, stream>>>(
            d_spheres, d_tile_offsets, d_tile_entity_pairs_sorted,
            d_lightTileList, d_tileClassifyCounts,
            d_depthBuffer, outputImage);
    }

    // Heavy tiles: persistent threads, 1 batch per work-steal (atomicMin depth)
    {
        unsigned int zero = 0;
        cudaMemcpyToSymbolAsync(g_nextHeavyWorkItem, &zero, sizeof(unsigned int), 0, cudaMemcpyHostToDevice, stream);
        heavySphereRasterKernel<<<grid, block, 0, stream>>>(
            d_spheres, d_tile_offsets, d_tile_entity_pairs_sorted,
            d_heavyTileIndices, d_heavyBatchStarts, d_tileClassifyCounts,
            d_depthBuffer, outputImage);
    }
}

extern "C" void launchTiledCylinderRasterBinning(
    const Cylinder* d_cylinders,
    const unsigned int* d_tile_offsets,
    const unsigned long long* d_tile_entity_pairs_sorted,
    unsigned int* d_depthBuffer,
    cudaSurfaceObject_t outputImage,
    unsigned int tilesX,
    unsigned int tilesY,
    unsigned int* d_lightTileList,
    unsigned int* d_heavyTileIndices,
    unsigned int* d_heavyBatchStarts,
    unsigned int* d_tileClassifyCounts,
    cudaStream_t stream)
{
    unsigned int totalTiles = tilesX * tilesY;

    cudaMemsetAsync(d_tileClassifyCounts, 0, 2 * sizeof(unsigned int), stream);
    {
        int threads = 256;
        int blocks = (totalTiles + threads - 1) / threads;
        classifyTilesKernel<<<blocks, threads, 0, stream>>>(
            d_tile_offsets, d_lightTileList, d_heavyTileIndices,
            d_heavyBatchStarts, d_tileClassifyCounts, totalTiles);
    }

    int numSMs = getNumSMs();
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(numSMs * 2, 1);

    {
        unsigned int zero = 0;
        cudaMemcpyToSymbolAsync(g_nextLightWorkItem, &zero, sizeof(unsigned int), 0, cudaMemcpyHostToDevice, stream);
        lightCylinderRasterKernel<<<grid, block, 0, stream>>>(
            d_cylinders, d_tile_offsets, d_tile_entity_pairs_sorted,
            d_lightTileList, d_tileClassifyCounts,
            d_depthBuffer, outputImage);
    }

    {
        unsigned int zero = 0;
        cudaMemcpyToSymbolAsync(g_nextHeavyWorkItem, &zero, sizeof(unsigned int), 0, cudaMemcpyHostToDevice, stream);
        heavyCylinderRasterKernel<<<grid, block, 0, stream>>>(
            d_cylinders, d_tile_offsets, d_tile_entity_pairs_sorted,
            d_heavyTileIndices, d_heavyBatchStarts, d_tileClassifyCounts,
            d_depthBuffer, outputImage);
    }
}
