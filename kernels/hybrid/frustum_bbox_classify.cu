/**
 * @file frustum_bbox_classify.cu
 * @brief Kernel 1: Frustum culling + BBox extraction + Size classification
 * 
 * This kernel performs:
 * 1. Frustum culling for all spheres/cylinders
 * 2. BBox 2D extraction for visible entities
 * 3. Classification into small (direct raster) vs large (tiled) entities
 * 4. Output to separate buffers for each path
 */

#define GLM_FORCE_CUDA
#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_INLINE

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "hybrid_types.h"
#include "geometry/cylinder/cylinder.h"

// Constant memory for per-frame data
__constant__ HybridConstants hybridCst;

// ============================================================================
// Device Helpers
// ============================================================================

__device__ inline bool isOnOrForwardOfPlane(const glm::vec4& plane, const glm::vec4& spherePosR) {
    float d = glm::dot(glm::vec3(plane), glm::vec3(spherePosR)) - plane.w;
    return d >= -spherePosR.w;
}

__device__ inline bool isSphereInsideFrustum(const glm::vec4& spherePosR) {
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        if (!isOnOrForwardOfPlane(hybridCst.frustumPlanes[i], spherePosR))
            return false;
    }
    return true;
}

__device__ inline bool isCylinderInsideFrustum(const glm::vec3& pa, const glm::vec3& pb, float radius) {
    // Test bounding sphere of cylinder
    glm::vec3 center = (pa + pb) * 0.5f;
    float halfLen = glm::length(pb - pa) * 0.5f;
    float boundingRadius = sqrtf(halfLen * halfLen + radius * radius);
    glm::vec4 boundingSphere = glm::vec4(center, boundingRadius);
    return isSphereInsideFrustum(boundingSphere);
}

__device__ inline void computeSphereBBox(
    const glm::vec4& spherePosR,
    glm::vec2& screenMin,
    glm::vec2& screenMax,
    float& centerDepth)
{
    // Transform to camera space
    glm::vec4 camSpace4 = hybridCst.view * glm::vec4(glm::vec3(spherePosR), 1.0f);
    glm::vec3 cameraSpaceSphere = glm::vec3(camSpace4);
    glm::vec3 normCamSpaceSphere = glm::normalize(cameraSpaceSphere);
    glm::vec3 camImposPos = cameraSpaceSphere - normCamSpaceSphere * spherePosR.w;
    
    // Compute billboard scale
    float dist = glm::length(cameraSpaceSphere) + 1e-6f;
    float sinAngle = spherePosR.w / dist;
    float tanAngle = tanf(asinf(fminf(sinAngle, 0.999f)));
    float quadScale = tanAngle * glm::length(camImposPos);
    
    // Compute billboard vectors
    glm::vec3 upVec = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 impU = glm::normalize(glm::cross(normCamSpaceSphere, upVec));
    if (glm::length(impU) < 0.001f) {
        impU = glm::vec3(1.0f, 0.0f, 0.0f);
    }
    glm::vec3 impV = glm::cross(impU, normCamSpaceSphere) * quadScale;
    impU *= quadScale;
    
    // Compute corners
    glm::vec3 corners[4] = {
        camImposPos + impU + impV,
        camImposPos - impU + impV,
        camImposPos + impU - impV,
        camImposPos - impU - impV
    };
    
    // Project and find min/max
    glm::vec2 minC(1e6f), maxC(-1e6f);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        glm::vec4 clip = hybridCst.proj * glm::vec4(corners[i], 1.0f);
        float iw = 1.0f / clip.w;
        float x = clip.x * iw;
        float y = clip.y * iw;
        minC.x = fminf(minC.x, x);
        minC.y = fminf(minC.y, y);
        maxC.x = fmaxf(maxC.x, x);
        maxC.y = fmaxf(maxC.y, y);
    }
    
    // Convert to screen coordinates
    screenMin.x = floorf((minC.x * 0.5f + 0.5f) * hybridCst.screenWidth);
    screenMin.y = floorf((minC.y * 0.5f + 0.5f) * hybridCst.screenHeight);
    screenMax.x = ceilf((maxC.x * 0.5f + 0.5f) * hybridCst.screenWidth);
    screenMax.y = ceilf((maxC.y * 0.5f + 0.5f) * hybridCst.screenHeight);
    
    // Clamp to screen
    screenMin.x = fmaxf(0.0f, screenMin.x);
    screenMin.y = fmaxf(0.0f, screenMin.y);
    screenMax.x = fminf((float)hybridCst.screenWidth, screenMax.x);
    screenMax.y = fminf((float)hybridCst.screenHeight, screenMax.y);
    
    // Compute center depth for sorting
    glm::vec4 centerClip = hybridCst.proj * glm::vec4(cameraSpaceSphere, 1.0f);
    centerDepth = centerClip.z / centerClip.w;
}

__device__ inline void computeCylinderBBox(
    const glm::vec3& pa,
    const glm::vec3& pb,
    float radius,
    glm::vec2& screenMin,
    glm::vec2& screenMax,
    float& centerDepth)
{
    // Transform endpoints to camera space
    glm::vec4 camA = hybridCst.view * glm::vec4(pa, 1.0f);
    glm::vec4 camB = hybridCst.view * glm::vec4(pb, 1.0f);
    
    // Compute axis and extent vectors
    glm::vec3 axis = glm::vec3(camB) - glm::vec3(camA);
    glm::vec3 e = radius * glm::sqrt(glm::vec3(1.0f) - axis * axis / glm::dot(axis, axis));
    
    // Compute 8 corners of the cylinder bounding box
    glm::vec3 corners[8] = {
        glm::vec3(camA) + e,
        glm::vec3(camA) - e,
        glm::vec3(camB) + e,
        glm::vec3(camB) - e,
        glm::vec3(camA) + glm::vec3(e.x, -e.y, e.z),
        glm::vec3(camA) + glm::vec3(-e.x, e.y, e.z),
        glm::vec3(camB) + glm::vec3(e.x, -e.y, e.z),
        glm::vec3(camB) + glm::vec3(-e.x, e.y, e.z)
    };
    
    // Project and find min/max
    glm::vec2 minC(1e6f), maxC(-1e6f);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        glm::vec4 clip = hybridCst.proj * glm::vec4(corners[i], 1.0f);
        if (clip.w > 0.001f) {
            float iw = 1.0f / clip.w;
            float x = clip.x * iw;
            float y = clip.y * iw;
            minC.x = fminf(minC.x, x);
            minC.y = fminf(minC.y, y);
            maxC.x = fmaxf(maxC.x, x);
            maxC.y = fmaxf(maxC.y, y);
        }
    }
    
    // Convert to screen coordinates
    screenMin.x = floorf((minC.x * 0.5f + 0.5f) * hybridCst.screenWidth);
    screenMin.y = floorf((minC.y * 0.5f + 0.5f) * hybridCst.screenHeight);
    screenMax.x = ceilf((maxC.x * 0.5f + 0.5f) * hybridCst.screenWidth);
    screenMax.y = ceilf((maxC.y * 0.5f + 0.5f) * hybridCst.screenHeight);
    
    // Clamp to screen
    screenMin.x = fmaxf(0.0f, screenMin.x);
    screenMin.y = fmaxf(0.0f, screenMin.y);
    screenMax.x = fminf((float)hybridCst.screenWidth, screenMax.x);
    screenMax.y = fminf((float)hybridCst.screenHeight, screenMax.y);
    
    // Compute center depth
    glm::vec3 center = (glm::vec3(camA) + glm::vec3(camB)) * 0.5f;
    glm::vec4 centerClip = hybridCst.proj * glm::vec4(center, 1.0f);
    centerDepth = centerClip.z / centerClip.w;
}

// ============================================================================
// Sphere Classification Kernel
// ============================================================================

__global__ void sphereFrustumBBoxClassifyKernel(
    const glm::vec4* __restrict__ spheres,
    // Output for small entities (direct raster)
    unsigned int* __restrict__ smallSphereIndices,
    unsigned int* __restrict__ smallSphereCount,
    // Output for large entities (tiled raster)
    SphereBillboard* __restrict__ largeBillboards,
    unsigned int* __restrict__ largeSphereCount,
    // Statistics
    unsigned int* __restrict__ frustumPassedCount)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hybridCst.sphereCount) return;
    
    glm::vec4 spherePosR = spheres[idx];
    
    // Frustum culling
    if (!isSphereInsideFrustum(spherePosR)) {
        return;
    }
    
    // Passed frustum culling
    if (hybridCst.benchmark) atomicAdd(frustumPassedCount, 1u);
    
    // Compute BBox
    glm::vec2 screenMin, screenMax;
    float centerDepth;
    computeSphereBBox(spherePosR, screenMin, screenMax, centerDepth);
    
    // Compute bbox area
    float width = screenMax.x - screenMin.x;
    float height = screenMax.y - screenMin.y;
    float area = width * height;
    
    // Skip tiny entities
    if (area < 1.0f) return;
    
    // Classify by size
    if (area <= (float)hybridCst.smallEntityThreshold) {
        // Small entity: add to direct raster list
        unsigned int outIdx = atomicAdd(smallSphereCount, 1u);
        smallSphereIndices[outIdx] = idx;
    } else {
        // Large entity: create billboard for tiled processing
        unsigned int outIdx = atomicAdd(largeSphereCount, 1u);
        
        SphereBillboard& bb = largeBillboards[outIdx];
        bb.positionRadius = spherePosR;
        bb.screenMin = screenMin;
        bb.screenMax = screenMax;
        bb.centerDepth = centerDepth;
        bb.originalIndex = idx;
        
        // Compute tile range
        bb.tileMinX = (unsigned int)(screenMin.x) >> TILE_SIZE_SHIFT;
        bb.tileMinY = (unsigned int)(screenMin.y) >> TILE_SIZE_SHIFT;
        bb.tileMaxX = (unsigned int)(screenMax.x - 1) >> TILE_SIZE_SHIFT;
        bb.tileMaxY = (unsigned int)(screenMax.y - 1) >> TILE_SIZE_SHIFT;
        
        // Clamp tile coords
        bb.tileMaxX = min(bb.tileMaxX, (unsigned int)(hybridCst.tilesX - 1));
        bb.tileMaxY = min(bb.tileMaxY, (unsigned int)(hybridCst.tilesY - 1));
    }
}

// ============================================================================
// Cylinder Classification Kernel
// ============================================================================

__global__ void cylinderFrustumBBoxClassifyKernel(
    const Cylinder* __restrict__ cylinders,
    // Output for small entities (direct raster)
    unsigned int* __restrict__ smallCylinderIndices,
    unsigned int* __restrict__ smallCylinderCount,
    // Output for large entities (tiled raster)
    CylinderBillboard* __restrict__ largeBillboards,
    unsigned int* __restrict__ largeCylinderCount,
    // Statistics
    unsigned int* __restrict__ frustumPassedCount)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hybridCst.cylinderCount) return;
    
    Cylinder cyl = cylinders[idx];
    glm::vec3 pa = glm::vec3(cyl.pa_r);
    glm::vec3 pb = glm::vec3(cyl.pb_r);
    float radius = cyl.pa_r.w;
    
    // Frustum culling
    if (!isCylinderInsideFrustum(pa, pb, radius)) {
        return;
    }
    
    // Passed frustum culling
    if (hybridCst.benchmark) atomicAdd(frustumPassedCount, 1u);
    
    // Compute BBox
    glm::vec2 screenMin, screenMax;
    float centerDepth;
    computeCylinderBBox(pa, pb, radius, screenMin, screenMax, centerDepth);
    
    // Compute bbox area
    float width = screenMax.x - screenMin.x;
    float height = screenMax.y - screenMin.y;
    float area = width * height;
    
    // Skip tiny entities
    if (area < 1.0f) return;
    
    // Classify by size
    if (area <= (float)hybridCst.smallEntityThreshold) {
        // Small entity: add to direct raster list
        unsigned int outIdx = atomicAdd(smallCylinderCount, 1u);
        smallCylinderIndices[outIdx] = idx;
    } else {
        // Large entity: create billboard for tiled processing
        unsigned int outIdx = atomicAdd(largeCylinderCount, 1u);
        
        CylinderBillboard& bb = largeBillboards[outIdx];
        bb.pa_r = cyl.pa_r;
        bb.pb_r = cyl.pb_r;
        bb.screenMin = screenMin;
        bb.screenMax = screenMax;
        bb.centerDepth = centerDepth;
        bb.originalIndex = idx;
        
        // Compute tile range
        bb.tileMinX = (unsigned int)(screenMin.x) >> TILE_SIZE_SHIFT;
        bb.tileMinY = (unsigned int)(screenMin.y) >> TILE_SIZE_SHIFT;
        bb.tileMaxX = (unsigned int)(screenMax.x - 1) >> TILE_SIZE_SHIFT;
        bb.tileMaxY = (unsigned int)(screenMax.y - 1) >> TILE_SIZE_SHIFT;
        
        // Clamp tile coords
        bb.tileMaxX = min(bb.tileMaxX, (unsigned int)(hybridCst.tilesX - 1));
        bb.tileMaxY = min(bb.tileMaxY, (unsigned int)(hybridCst.tilesY - 1));
    }
}

// ============================================================================
// Host-side wrapper functions
// ============================================================================

extern "C" void uploadHybridConstants(const HybridConstants& constants, cudaStream_t stream) {
    cudaMemcpyToSymbolAsync(hybridCst, &constants, sizeof(HybridConstants), 0, cudaMemcpyHostToDevice, stream);
}

extern "C" void launchSphereFrustumBBoxClassify(
    const glm::vec4* d_spheres,
    unsigned int* d_smallSphereIndices,
    unsigned int* d_smallSphereCount,
    SphereBillboard* d_largeBillboards,
    unsigned int* d_largeSphereCount,
    unsigned int* d_frustumPassedCount,
    int sphereCount,
    cudaStream_t stream)
{
    dim3 block(CULL_BLOCK_SIZE);
    dim3 grid((sphereCount + block.x - 1) / block.x);
    
    sphereFrustumBBoxClassifyKernel<<<grid, block, 0, stream>>>(
        d_spheres,
        d_smallSphereIndices,
        d_smallSphereCount,
        d_largeBillboards,
        d_largeSphereCount,
        d_frustumPassedCount
    );
}

extern "C" void launchCylinderFrustumBBoxClassify(
    const Cylinder* d_cylinders,
    unsigned int* d_smallCylinderIndices,
    unsigned int* d_smallCylinderCount,
    CylinderBillboard* d_largeBillboards,
    unsigned int* d_largeCylinderCount,
    unsigned int* d_frustumPassedCount,
    int cylinderCount,
    cudaStream_t stream)
{
    dim3 block(CULL_BLOCK_SIZE);
    dim3 grid((cylinderCount + block.x - 1) / block.x);
    
    cylinderFrustumBBoxClassifyKernel<<<grid, block, 0, stream>>>(
        d_cylinders,
        d_smallCylinderIndices,
        d_smallCylinderCount,
        d_largeBillboards,
        d_largeCylinderCount,
        d_frustumPassedCount
    );
}
