/**
 * @file tiled_raster_binning.cu
 * @brief Work-group expansion from RLE + unified tiled rasterization.
 * After sort+RLE, each tile's entities are split into work groups of SHARED_BATCH.
 * Each CUDA block processes one work group: loads entities into shared memory,
 * ray-traces, and uses atomicMin on the depth buffer.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>
#include "hybrid_binning_types.h"
#include "geometry/cylinder/cylinder.h"

extern __constant__ HybridConstants hybridCst;

#define TILE_SIZE 16
#define SHARED_BATCH 64

// ─────────── Helpers ───────────

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

// ─────────── Work-group expansion from RLE results ───────────

/**
 * Expand RLE runs into a flat work-group dispatch list.
 * Each RLE run i represents a tile with d_counts_out[i] entities starting
 * at d_run_offsets[i] in the sorted pairs array.
 * We split each run into ceil(count/SHARED_BATCH) work groups.
 * Output: d_wg_tileId, d_wg_entityStart, d_wg_entityCount (one per WG).
 * d_wg_totalCount is atomically incremented to give the total WG count.
 */
__global__ void expandWorkGroupsKernel(
    const unsigned int* __restrict__ d_unique_out,
    const unsigned int* __restrict__ d_counts_out,
    const unsigned int* __restrict__ d_run_offsets,
    unsigned int numRuns,
    unsigned int* __restrict__ d_wg_tileId,
    unsigned int* __restrict__ d_wg_entityStart,
    unsigned int* __restrict__ d_wg_entityCount,
    unsigned int* __restrict__ d_wg_totalCount)
{
    unsigned int runIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (runIdx >= numRuns) return;

    unsigned int tileId    = d_unique_out[runIdx];
    unsigned int count     = d_counts_out[runIdx];
    unsigned int runStart  = d_run_offsets[runIdx];
    unsigned int numWG     = (count + SHARED_BATCH - 1) / SHARED_BATCH;

    unsigned int baseSlot = atomicAdd(d_wg_totalCount, numWG);

    for (unsigned int g = 0; g < numWG; g++) {
        unsigned int wgStart = g * SHARED_BATCH;
        unsigned int wgCount = min((unsigned int)SHARED_BATCH, count - wgStart);
        d_wg_tileId    [baseSlot + g] = tileId;
        d_wg_entityStart[baseSlot + g] = runStart + wgStart;
        d_wg_entityCount[baseSlot + g] = wgCount;
    }
}

__device__ inline void writeFragmentSafe(
    unsigned int* depthBuffer, int pixelIdx,
    float depth, const glm::vec3& color,
    cudaSurfaceObject_t outputImage, int px, int py)
{
    unsigned int newDepth = __float_as_uint(depth);
    unsigned int old = depthBuffer[pixelIdx];

    while (__uint_as_float(old) > depth) {
        unsigned int assumed = old;
        old = atomicCAS(&depthBuffer[pixelIdx], assumed, newDepth);
        if (old == assumed) {
            uchar4 pixel = make_uchar4(
                (unsigned char)(color.x * 255.0f),
                (unsigned char)(color.y * 255.0f),
                (unsigned char)(color.z * 255.0f), 255);
            surf2Dwrite(pixel, outputImage, px * sizeof(uchar4), py);
            return;
        }
    }
}


// ─────────── Sphere tiled raster (one block per work group) ───────────

__global__ void tiledSphereRasterWGKernel(
    const glm::vec4* __restrict__ d_spheres,
    const unsigned long long* __restrict__ d_tile_entity_pairs_sorted,
    const unsigned int* __restrict__ d_wg_tileId,
    const unsigned int* __restrict__ d_wg_entityStart,
    const unsigned int* __restrict__ d_wg_entityCount,
    unsigned int* __restrict__ depthBuffer,
    cudaSurfaceObject_t outputImage)
{
    const float aspectRatio = (float)hybridCst.screenWidth / (float)hybridCst.screenHeight;
    const float fovRad = glm::radians(hybridCst.fov);
    const float fovTan = tanf(fovRad * 0.5f);
    const float halfFovTan = fovTan * aspectRatio;
    const float proj22 = hybridCst.proj[2][2];
    const float proj32 = hybridCst.proj[3][2];

    __shared__ glm::vec4 shPosR[SHARED_BATCH];

    const unsigned int wgIdx       = blockIdx.x;
    const unsigned int tileIdx     = d_wg_tileId[wgIdx];
    const unsigned int entityStart = d_wg_entityStart[wgIdx];
    const unsigned int entityCount = d_wg_entityCount[wgIdx];

    const unsigned int tileX  = tileIdx % hybridCst.tilesX;
    const unsigned int tileY  = tileIdx / hybridCst.tilesX;
    const int pixelX = tileX * TILE_SIZE + threadIdx.x;
    const int pixelY = tileY * TILE_SIZE + threadIdx.y;
    const bool validPixel = (pixelX < hybridCst.screenWidth && pixelY < hybridCst.screenHeight);

    bool hasHit = false;
    glm::vec3 finalColor = glm::vec3(0.0f);
    float finalDepth = __uint_as_float(depthBuffer[pixelY * hybridCst.screenWidth + pixelX]);

    // Load entities into shared memory
    const unsigned int localIdx = threadIdx.y * TILE_SIZE + threadIdx.x;
    if (localIdx < SHARED_BATCH && localIdx < entityCount) {
        unsigned long long pair = d_tile_entity_pairs_sorted[entityStart + localIdx];
        unsigned int sphereIndex = (unsigned int)(pair & 0xFFFFFFFFu);
        shPosR[localIdx] = d_spheres[sphereIndex];
    }
    __syncthreads();

    // Ray-trace; atomicMin since multiple blocks may share a tile
    if (!validPixel) return;

    glm::vec3 rd  = computeRayDirTiled(pixelX, pixelY, fovTan, halfFovTan);
    glm::vec3 ray = hybridCst.rayStart + (float)pixelX * hybridCst.dx + (float)pixelY * hybridCst.dy;
    const int pixelIdx = pixelY * hybridCst.screenWidth + pixelX;

    for (unsigned int i = 0; i < entityCount; i++) {
        glm::vec4 posR = shPosR[i];
        float t = iSphereTiled(hybridCst.cameraPos, rd, glm::vec3(posR), posR.w);
        if (t > 0.0f) {
            glm::vec3 hit = ray * t;
            float depth = __fdividef(fmaf(hit.z, proj22, proj32), -hit.z);
            
            if (depth < finalDepth) {
                hasHit = true;
                glm::vec3 worldHit = hybridCst.cameraPos + rd * t;
                glm::vec3 normal = glm::normalize(worldHit - glm::vec3(posR));
                float lambert = fmaxf(0.0f, glm::dot(normal, -glm::normalize(rd * t)));
                finalColor = glm::vec3(atomsColor[0], atomsColor[1], atomsColor[2]) * lambert * diffuse;
                finalDepth = depth;
            }
        }
    }

    if (hasHit)
    {
            writeFragmentSafe(depthBuffer, pixelIdx, finalDepth, finalColor,
                          outputImage, pixelX, pixelY);
    }
    
}

// ─────────── Cylinder tiled raster (one block per work group) ───────────

__global__ void tiledCylinderRasterWGKernel(
    const Cylinder* __restrict__ d_cylinders,
    const unsigned long long* __restrict__ d_tile_entity_pairs_sorted,
    const unsigned int* __restrict__ d_wg_tileId,
    const unsigned int* __restrict__ d_wg_entityStart,
    const unsigned int* __restrict__ d_wg_entityCount,
    unsigned int* __restrict__ depthBuffer,
    cudaSurfaceObject_t outputImage)
{
    const float aspectRatio = (float)hybridCst.screenWidth / (float)hybridCst.screenHeight;
    const float fovRad = glm::radians(hybridCst.fov);
    const float fovTan = tanf(fovRad * 0.5f);
    const float halfFovTan = fovTan * aspectRatio;
    const float proj22 = hybridCst.proj[2][2];
    const float proj32 = hybridCst.proj[3][2];

    
    __shared__ glm::vec4 shPa[SHARED_BATCH], shPb[SHARED_BATCH];

    const unsigned int wgIdx       = blockIdx.x;
    const unsigned int tileIdx     = d_wg_tileId[wgIdx];
    const unsigned int entityStart = d_wg_entityStart[wgIdx];
    const unsigned int entityCount = d_wg_entityCount[wgIdx];

    const unsigned int tileX  = tileIdx % hybridCst.tilesX;
    const unsigned int tileY  = tileIdx / hybridCst.tilesX;
    const int pixelX = tileX * TILE_SIZE + threadIdx.x;
    const int pixelY = tileY * TILE_SIZE + threadIdx.y;
    const bool validPixel = (pixelX < hybridCst.screenWidth && pixelY < hybridCst.screenHeight);
    
    bool hasHit = false;
    glm::vec3 finalColor = glm::vec3(0.0f);
    float finalDepth = __uint_as_float(depthBuffer[pixelY * hybridCst.screenWidth + pixelX]); 

    // Load entities into shared memory
    const unsigned int localIdx = threadIdx.y * TILE_SIZE + threadIdx.x;
    if (localIdx < SHARED_BATCH && localIdx < entityCount) {
        unsigned long long pair = d_tile_entity_pairs_sorted[entityStart + localIdx];
        unsigned int cylIndex = (unsigned int)(pair & 0xFFFFFFFFu);
        Cylinder cyl = d_cylinders[cylIndex];
        shPa[localIdx] = cyl.pa_r;
        shPb[localIdx] = cyl.pb_r;
    }
    __syncthreads();

    // Ray-trace; atomicMin since multiple blocks may share a tile
    if (!validPixel) return;

    glm::vec3 rd  = computeRayDirTiled(pixelX, pixelY, fovTan, halfFovTan);
    glm::vec3 ray = hybridCst.rayStart + (float)pixelX * hybridCst.dx + (float)pixelY * hybridCst.dy;
    const int pixelIdx = pixelY * hybridCst.screenWidth + pixelX;

    for (unsigned int i = 0; i < entityCount; i++) {
        glm::vec3 pa = glm::vec3(shPa[i]), pb = glm::vec3(shPb[i]);
        float ra = shPa[i].w;
        glm::vec4 tnor = iCylinderTiled(hybridCst.cameraPos, rd, pa, pb, ra);
        if (tnor.x > 0.0f) {
            float t = tnor.x;
            glm::vec3 hit = ray * t;
            float depth = __fdividef(fmaf(hit.z, proj22, proj32), -hit.z);
            
            if (depth < finalDepth) {
                hasHit = true;
                glm::vec3 normal = glm::normalize(glm::vec3(tnor.y, tnor.z, tnor.w));
                float lambert = glm::max(0.0f, glm::dot(normal, -glm::normalize(rd * t)));
                finalColor = glm::vec3(bondsColor[0], bondsColor[1], bondsColor[2]) * lambert * diffuse;
                finalDepth = depth;
            }
        }
    }

    if (hasHit)
    { 
        writeFragmentSafe(depthBuffer, pixelIdx, finalDepth, finalColor,
                        outputImage, pixelX, pixelY);   
    }
    
}

// ─────────── Launch wrappers (extern "C") ───────────

extern "C" void launchExpandWorkGroups(
    const unsigned int* d_unique_out,
    const unsigned int* d_counts_out,
    const unsigned int* d_run_offsets,
    unsigned int numRuns,
    unsigned int* d_wg_tileId,
    unsigned int* d_wg_entityStart,
    unsigned int* d_wg_entityCount,
    unsigned int* d_wg_totalCount,
    cudaStream_t stream)
{
    if (numRuns == 0) return;
    int threads = 256;
    int blocks = (numRuns + threads - 1) / threads;
    expandWorkGroupsKernel<<<blocks, threads, 0, stream>>>(
        d_unique_out, d_counts_out, d_run_offsets, numRuns,
        d_wg_tileId, d_wg_entityStart, d_wg_entityCount, d_wg_totalCount);
}

extern "C" void launchTiledSphereRasterWG(
    const glm::vec4* d_spheres,
    const unsigned long long* d_tile_entity_pairs_sorted,
    const unsigned int* d_wg_tileId,
    const unsigned int* d_wg_entityStart,
    const unsigned int* d_wg_entityCount,
    unsigned int totalWorkGroups,
    unsigned int* d_depthBuffer,
    cudaSurfaceObject_t outputImage,
    cudaStream_t stream)
{
    if (totalWorkGroups == 0) return;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(totalWorkGroups, 1);
    tiledSphereRasterWGKernel<<<grid, block, 0, stream>>>(
        d_spheres, d_tile_entity_pairs_sorted,
        d_wg_tileId, d_wg_entityStart, d_wg_entityCount,
        d_depthBuffer, outputImage);
}

extern "C" void launchTiledCylinderRasterWG(
    const Cylinder* d_cylinders,
    const unsigned long long* d_tile_entity_pairs_sorted,
    const unsigned int* d_wg_tileId,
    const unsigned int* d_wg_entityStart,
    const unsigned int* d_wg_entityCount,
    unsigned int totalWorkGroups,
    unsigned int* d_depthBuffer,
    cudaSurfaceObject_t outputImage,
    cudaStream_t stream)
{
    if (totalWorkGroups == 0) return;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(totalWorkGroups, 1);
    tiledCylinderRasterWGKernel<<<grid, block, 0, stream>>>(
        d_cylinders, d_tile_entity_pairs_sorted,
        d_wg_tileId, d_wg_entityStart, d_wg_entityCount,
        d_depthBuffer, outputImage);
}
