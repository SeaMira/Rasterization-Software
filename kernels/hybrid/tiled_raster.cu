/**
 * @file tiled_raster.cu
 * @brief Kernel 4b: Tiled rasterization for large entities (1 thread per pixel per tile)
 * 
 * Uses the second version approach with tile-based processing.
 * Each tile's thread block processes only entities assigned to that tile.
 * Entities are sorted front-to-back for early-Z rejection.
 */

#include <cuda_runtime.h>
#include <cuda.h>  // Required for CUDA_VERSION definition before GLM
#include <device_launch_parameters.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "hybrid_types.h"
#include "geometry/cylinder/cylinder.h"

// Access to constants
extern __constant__ HybridConstants hybridCst;

// Shared memory for loading billboards
#define SHARED_BILLBOARD_BATCH 64

// ============================================================================
// Ray intersection functions (duplicated for kernel isolation)
// ============================================================================

__device__ inline float iSphereTiled(const glm::vec3& ro, const glm::vec3& rd, 
                                      const glm::vec3& center, float radius) {
    glm::vec3 oc = ro - center;
    float b = glm::dot(oc, rd);
    float c = glm::dot(oc, oc) - radius * radius;
    float h = b * b - c;
    if (h < 0.0f) return -1.0f;
    return -b - sqrtf(h);
}

__device__ inline glm::vec2 iCylinderTiled(const glm::vec3& ro, const glm::vec3& rd,
                                            const glm::vec3& pa, const glm::vec3& pb, float ra) {
    glm::vec3 ba = pb - pa;
    glm::vec3 oc = ro - pa;
    
    float baba = glm::dot(ba, ba);
    float bard = glm::dot(ba, rd);
    float baoc = glm::dot(ba, oc);
    
    float k2 = baba - bard * bard;
    float k1 = baba * glm::dot(oc, rd) - baoc * bard;
    float k0 = baba * glm::dot(oc, oc) - baoc * baoc - ra * ra * baba;
    
    float h = k1 * k1 - k2 * k0;
    if (h < 0.0f) return glm::vec2(-1.0f);
    
    h = sqrtf(h);
    float t = (-k1 - h) / k2;
    
    float y = baoc + t * bard;
    if (y > 0.0f && y < baba) {
        return glm::vec2(t, y / baba);
    }
    
    t = ((y < 0.0f ? 0.0f : baba) - baoc) / bard;
    if (fabsf(k1 + k2 * t) < h) {
        return glm::vec2(t, y < 0.0f ? 0.0f : 1.0f);
    }
    
    return glm::vec2(-1.0f);
}

__device__ inline glm::vec3 computeRayDirTiled(int px, int py, float fovTan, float halfFovTan) {
    glm::vec2 p = (-glm::vec2(hybridCst.screenWidth, hybridCst.screenHeight) + 
                   2.0f * glm::vec2(px, py)) / 
                   glm::vec2(hybridCst.screenWidth, hybridCst.screenHeight);
    return glm::normalize(p.x * hybridCst.right * halfFovTan + 
                          p.y * hybridCst.up * fovTan + 
                          hybridCst.front);
}

// ============================================================================
// Tiled Sphere Rasterization Kernel
// ============================================================================

__global__ void tiledSphereRasterKernel(
    const SphereBillboard* __restrict__ billboards,
    const unsigned int* __restrict__ tileOffsets,
    const unsigned int* __restrict__ tileCounts,
    const unsigned int* __restrict__ tileEntityIndices,
    unsigned int* __restrict__ depthBuffer,
    cudaSurfaceObject_t outputImage)
{
    // Tile coordinates
    const unsigned int tileX = blockIdx.x;
    const unsigned int tileY = blockIdx.y;
    const unsigned int tileIdx = tileY * hybridCst.tilesX + tileX;
    
    // Pixel coordinates within tile
    const unsigned int localX = threadIdx.x;
    const unsigned int localY = threadIdx.y;
    const int pixelX = tileX * TILE_SIZE + localX;
    const int pixelY = tileY * TILE_SIZE + localY;
    
    // Check bounds
    if (pixelX >= hybridCst.screenWidth || pixelY >= hybridCst.screenHeight) return;
    
    // Get tile entity info
    const unsigned int entityOffset = tileOffsets[tileIdx];
    const unsigned int entityCount = tileCounts[tileIdx];
    
    if (entityCount == 0) return;
    
    // Shared memory for batch loading billboards
    __shared__ SphereBillboard sharedBillboards[SHARED_BILLBOARD_BATCH];
    
    // Per-thread state
    float minDepth = 1e10f;
    glm::vec3 finalColor = glm::vec3(0.0f);
    bool hasHit = false;
    
    // Ray casting setup
    float aspectRatio = (float)hybridCst.screenWidth / (float)hybridCst.screenHeight;
    float fovRad = glm::radians(hybridCst.fov);
    float fovTan = tanf(fovRad * 0.5f);
    float halfFovTan = fovTan * aspectRatio;
    glm::vec3 rd = computeRayDirTiled(pixelX, pixelY, fovTan, halfFovTan);
    
    // Process entities in batches
    unsigned int numBatches = (entityCount + SHARED_BILLBOARD_BATCH - 1) / SHARED_BILLBOARD_BATCH;
    
    for (unsigned int batch = 0; batch < numBatches; ++batch) {
        // Cooperative load into shared memory
        unsigned int localIdx = threadIdx.y * TILE_SIZE + threadIdx.x;
        unsigned int batchStart = batch * SHARED_BILLBOARD_BATCH;
        
        if (localIdx < SHARED_BILLBOARD_BATCH) {
            unsigned int entityIdx = batchStart + localIdx;
            if (entityIdx < entityCount) {
                unsigned int billboardIdx = tileEntityIndices[entityOffset + entityIdx];
                sharedBillboards[localIdx] = billboards[billboardIdx];
            }
        }
        __syncthreads();
        
        // Process this batch
        unsigned int batchCount = min(SHARED_BILLBOARD_BATCH, entityCount - batchStart);
        
        for (unsigned int i = 0; i < batchCount; ++i) {
            const SphereBillboard& bb = sharedBillboards[i];
            
            // Early-Z: if we already have a closer hit, skip entities that are further
            // (entities are sorted front-to-back)
            if (hasHit && bb.centerDepth > minDepth) {
                continue;
            }
            
            // Check if pixel is in this billboard's bbox
            if (pixelX < bb.screenMin.x || pixelX >= bb.screenMax.x ||
                pixelY < bb.screenMin.y || pixelY >= bb.screenMax.y) {
                continue;
            }
            
            // Ray-sphere intersection
            glm::vec3 spherePos = glm::vec3(bb.positionRadius);
            float radius = bb.positionRadius.w;
            float t = iSphereTiled(hybridCst.cameraPos, rd, spherePos, radius);
            
            if (t > 0.0f) {
                glm::vec3 hit = hybridCst.cameraPos + rd * t;
                glm::vec4 hitClip = hybridCst.proj * (hybridCst.view * glm::vec4(hit, 1.0f));
                float depth = hitClip.z / hitClip.w;
                
                if (depth < minDepth) {
                    minDepth = depth;
                    hasHit = true;
                    
                    // Compute shading
                    glm::vec3 normal = glm::normalize(hit - spherePos);
                    float lambert = fmaxf(0.0f, glm::dot(normal, -glm::normalize(rd)));
                    finalColor = glm::vec3(0.01f, 1.0f, 0.05f) * lambert * 0.9f;
                }
            }
        }
        __syncthreads();
    }
    
    // Write result
    if (hasHit) {
        int pixelIdx = pixelY * hybridCst.screenWidth + pixelX;
        unsigned int depthU = __float_as_uint(minDepth);
        
        unsigned int old = atomicMin(&depthBuffer[pixelIdx], depthU);
        if (__uint_as_float(old) > minDepth) {
            uchar4 pixel = make_uchar4(
                (unsigned char)(finalColor.x * 255.0f),
                (unsigned char)(finalColor.y * 255.0f),
                (unsigned char)(finalColor.z * 255.0f),
                255);
            surf2Dwrite(pixel, outputImage, pixelX * sizeof(uchar4), pixelY);
        }
    }
}

// ============================================================================
// Tiled Cylinder Rasterization Kernel
// ============================================================================

__global__ void tiledCylinderRasterKernel(
    const CylinderBillboard* __restrict__ billboards,
    const unsigned int* __restrict__ tileOffsets,
    const unsigned int* __restrict__ tileCounts,
    const unsigned int* __restrict__ tileEntityIndices,
    unsigned int* __restrict__ depthBuffer,
    cudaSurfaceObject_t outputImage)
{
    const unsigned int tileX = blockIdx.x;
    const unsigned int tileY = blockIdx.y;
    const unsigned int tileIdx = tileY * hybridCst.tilesX + tileX;
    
    const unsigned int localX = threadIdx.x;
    const unsigned int localY = threadIdx.y;
    const int pixelX = tileX * TILE_SIZE + localX;
    const int pixelY = tileY * TILE_SIZE + localY;
    
    if (pixelX >= hybridCst.screenWidth || pixelY >= hybridCst.screenHeight) return;
    
    const unsigned int entityOffset = tileOffsets[tileIdx];
    const unsigned int entityCount = tileCounts[tileIdx];
    
    if (entityCount == 0) return;
    
    __shared__ CylinderBillboard sharedBillboards[SHARED_BILLBOARD_BATCH];
    
    float minDepth = 1e10f;
    glm::vec3 finalColor = glm::vec3(0.0f);
    bool hasHit = false;
    
    float aspectRatio = (float)hybridCst.screenWidth / (float)hybridCst.screenHeight;
    float fovRad = glm::radians(hybridCst.fov);
    float fovTan = tanf(fovRad * 0.5f);
    float halfFovTan = fovTan * aspectRatio;
    glm::vec3 rd = computeRayDirTiled(pixelX, pixelY, fovTan, halfFovTan);
    
    unsigned int numBatches = (entityCount + SHARED_BILLBOARD_BATCH - 1) / SHARED_BILLBOARD_BATCH;
    
    for (unsigned int batch = 0; batch < numBatches; ++batch) {
        unsigned int localIdx = threadIdx.y * TILE_SIZE + threadIdx.x;
        unsigned int batchStart = batch * SHARED_BILLBOARD_BATCH;
        
        if (localIdx < SHARED_BILLBOARD_BATCH) {
            unsigned int entityIdx = batchStart + localIdx;
            if (entityIdx < entityCount) {
                unsigned int billboardIdx = tileEntityIndices[entityOffset + entityIdx];
                sharedBillboards[localIdx] = billboards[billboardIdx];
            }
        }
        __syncthreads();
        
        unsigned int batchCount = min(SHARED_BILLBOARD_BATCH, entityCount - batchStart);
        
        for (unsigned int i = 0; i < batchCount; ++i) {
            const CylinderBillboard& bb = sharedBillboards[i];
            
            if (hasHit && bb.centerDepth > minDepth) {
                continue;
            }
            
            if (pixelX < bb.screenMin.x || pixelX >= bb.screenMax.x ||
                pixelY < bb.screenMin.y || pixelY >= bb.screenMax.y) {
                continue;
            }
            
            glm::vec3 pa = glm::vec3(bb.pa_r);
            glm::vec3 pb = glm::vec3(bb.pb_r);
            float radius = bb.pa_r.w;
            
            glm::vec2 result = iCylinderTiled(hybridCst.cameraPos, rd, pa, pb, radius);
            
            if (result.x > 0.0f) {
                glm::vec3 hit = hybridCst.cameraPos + rd * result.x;
                glm::vec4 hitClip = hybridCst.proj * (hybridCst.view * glm::vec4(hit, 1.0f));
                float depth = hitClip.z / hitClip.w;
                
                if (depth < minDepth) {
                    minDepth = depth;
                    hasHit = true;
                    
                    glm::vec3 ba = pb - pa;
                    glm::vec3 hitLocal = hit - pa;
                    float h = glm::dot(hitLocal, ba) / glm::dot(ba, ba);
                    glm::vec3 normal;
                    if (h > 0.0f && h < 1.0f) {
                        normal = glm::normalize(hitLocal - ba * h);
                    } else {
                        normal = glm::normalize(hitLocal - ba * glm::clamp(h, 0.0f, 1.0f));
                    }
                    
                    float lambert = fmaxf(0.0f, glm::dot(normal, -glm::normalize(rd)));
                    finalColor = glm::vec3(0.01f, 1.0f, 0.05f) * lambert * 0.9f;
                }
            }
        }
        __syncthreads();
    }
    
    if (hasHit) {
        int pixelIdx = pixelY * hybridCst.screenWidth + pixelX;
        unsigned int depthU = __float_as_uint(minDepth);
        
        unsigned int old = atomicMin(&depthBuffer[pixelIdx], depthU);
        if (__uint_as_float(old) > minDepth) {
            uchar4 pixel = make_uchar4(
                (unsigned char)(finalColor.x * 255.0f),
                (unsigned char)(finalColor.y * 255.0f),
                (unsigned char)(finalColor.z * 255.0f),
                255);
            surf2Dwrite(pixel, outputImage, pixelX * sizeof(uchar4), pixelY);
        }
    }
}

// ============================================================================
// Host wrappers
// ============================================================================

// Forward declaration of TileBinningData
struct TileBinningData;

extern "C" void launchTiledSphereRaster(
    const SphereBillboard* d_billboards,
    const unsigned int* d_tileOffsets,
    const unsigned int* d_tileCounts,
    const unsigned int* d_tileEntityIndices,
    unsigned int* d_depthBuffer,
    cudaSurfaceObject_t outputImage,
    unsigned int tilesX,
    unsigned int tilesY,
    cudaStream_t stream)
{
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(tilesX, tilesY);
    
    tiledSphereRasterKernel<<<grid, block, 0, stream>>>(
        d_billboards, d_tileOffsets, d_tileCounts, d_tileEntityIndices,
        d_depthBuffer, outputImage);
}

extern "C" void launchTiledCylinderRaster(
    const CylinderBillboard* d_billboards,
    const unsigned int* d_tileOffsets,
    const unsigned int* d_tileCounts,
    const unsigned int* d_tileEntityIndices,
    unsigned int* d_depthBuffer,
    cudaSurfaceObject_t outputImage,
    unsigned int tilesX,
    unsigned int tilesY,
    cudaStream_t stream)
{
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(tilesX, tilesY);
    
    tiledCylinderRasterKernel<<<grid, block, 0, stream>>>(
        d_billboards, d_tileOffsets, d_tileCounts, d_tileEntityIndices,
        d_depthBuffer, outputImage);
}
