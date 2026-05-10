/**
 * @file frustum_bbox_classify.cu
 * @brief Frustum culling + BBox + Size classification (same as hybrid, for binning pipeline).
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>

#include "hybrid_binning_types.h"
#include "geometry/cylinder/cylinder.h"

__constant__ HybridConstants hybridCst;

__device__ inline bool isOnOrForwardOfPlane(const glm::vec4& plane, const glm::vec4& spherePosR) {
    float d = glm::dot(glm::vec3(plane), glm::vec3(spherePosR)) - plane.w;
    return d >= -spherePosR.w;
}

__device__ inline bool isOnOrForwardPlaneAABB(glm::vec4 plane, _BBox3D& bbox) 
{
    glm::vec3 negativeVertex = bbox.mMin;
    glm::vec3 normal = glm::vec3(plane.x, plane.y, plane.z);
    float distance = plane.w;

    if (normal.x >= 0) negativeVertex.x = bbox.mMax.x;
        else negativeVertex.x = bbox.mMin.x;

    if (normal.y >= 0) negativeVertex.y = bbox.mMax.y;
        else negativeVertex.y = bbox.mMin.y;

    if (normal.z >= 0) negativeVertex.z = bbox.mMax.z;
        else negativeVertex.z = bbox.mMin.z;

    return glm::dot(normal, negativeVertex) - distance >= 0;
}

__device__ inline bool isSphereInsideFrustum(const glm::vec4& spherePosR) {
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        if (!isOnOrForwardOfPlane(hybridCst.frustumPlanes[i], spherePosR))
            return false;
    }
    return true;
}

__device__ inline bool isCylinderInsideFrustum(const glm::vec3 pa,
    const glm::vec3 pb, float radius)
{
    glm::vec3 a = pb - pa;
    glm::vec3 e = radius*sqrt( 1.0f - a*a/glm::dot(a,a) );
    glm::vec3 pa_sub_e = pa - e;
    glm::vec3 pb_sub_e = pb - e;
    glm::vec3 pa_add_e = pa + e;
    glm::vec3 pb_add_e = pb + e;
    _BBox3D bbox3d = _BBox3D{glm::vec3(fminf(pa_sub_e.x, pb_sub_e.x), fminf(pa_sub_e.y, pb_sub_e.y), fminf(pa_sub_e.z, pb_sub_e.z)), glm::vec3(fmaxf(pa_add_e.x, pb_add_e.x), fmaxf(pa_add_e.y, pb_add_e.y), fmaxf(pa_add_e.z, pb_add_e.z))};
    return isOnOrForwardPlaneAABB(hybridCst.frustumPlanes[0], bbox3d) &&
    isOnOrForwardPlaneAABB(hybridCst.frustumPlanes[1], bbox3d) &&
    isOnOrForwardPlaneAABB(hybridCst.frustumPlanes[2], bbox3d) &&
    isOnOrForwardPlaneAABB(hybridCst.frustumPlanes[3], bbox3d) &&
    isOnOrForwardPlaneAABB(hybridCst.frustumPlanes[4], bbox3d) &&
    isOnOrForwardPlaneAABB(hybridCst.frustumPlanes[5], bbox3d);
}

__device__ inline void computeSphereBBox(
    const glm::vec4& spherePosR,
    glm::vec2& screenMin,
    glm::vec2& screenMax,
    float& centerDepth)
{
    glm::vec4 camSpace4 = hybridCst.view * glm::vec4(glm::vec3(spherePosR), 1.0f);
    glm::vec3 cameraSpaceSphere = glm::vec3(camSpace4);
    glm::vec3 normCamSpaceSphere = glm::normalize(cameraSpaceSphere);
    glm::vec3 camImposPos = cameraSpaceSphere - normCamSpaceSphere * spherePosR.w;
    
    float dist = glm::length(cameraSpaceSphere) + 1e-6f;
    float sinAngle = spherePosR.w / dist;
    float tanAngle = tanf(asinf(fminf(sinAngle, 0.999f)));
    float quadScale = tanAngle * glm::length(camImposPos);
    
    glm::vec3 upVec = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 impU = glm::normalize(glm::cross(normCamSpaceSphere, upVec));
    if (glm::length(impU) < 0.001f) impU = glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 impV = glm::cross(impU, normCamSpaceSphere) * quadScale;
    impU *= quadScale;
    
    glm::vec3 corners[4] = {
        camImposPos + impU + impV,
        camImposPos - impU + impV,
        camImposPos + impU - impV,
        camImposPos - impU - impV
    };
    
    glm::vec2 minC(1e6f), maxC(-1e6f);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        glm::vec4 clip = hybridCst.proj * glm::vec4(corners[i], 1.0f);
        float iw = 1.0f / clip.w;
        minC.x = fminf(minC.x, clip.x * iw);
        minC.y = fminf(minC.y, clip.y * iw);
        maxC.x = fmaxf(maxC.x, clip.x * iw);
        maxC.y = fmaxf(maxC.y, clip.y * iw);
    }
    
    screenMin.x = floorf((minC.x * 0.5f + 0.5f) * hybridCst.screenWidth);
    screenMin.y = floorf((minC.y * 0.5f + 0.5f) * hybridCst.screenHeight);
    screenMax.x = ceilf((maxC.x * 0.5f + 0.5f) * hybridCst.screenWidth);
    screenMax.y = ceilf((maxC.y * 0.5f + 0.5f) * hybridCst.screenHeight);
    
    screenMin.x = fmaxf(0.0f, screenMin.x);
    screenMin.y = fmaxf(0.0f, screenMin.y);
    screenMax.x = fminf((float)hybridCst.screenWidth, screenMax.x);
    screenMax.y = fminf((float)hybridCst.screenHeight, screenMax.y);
    
    glm::vec4 centerClip = hybridCst.proj * glm::vec4(cameraSpaceSphere, 1.0f);
    centerDepth = centerClip.z / centerClip.w;
}

/**
 * Compute the screen-space AABB of a cylinder from an 8-corner camera-space
 * AABB. Out-parameters:
 *   `allCornersValid` is set to false when at least one of the 8 corners has
 *   `clip.w <= eps` (i.e. the cylinder straddles the near plane). In that
 *   case the returned AABB is built only from the valid corners, so the
 *   caller MUST treat it as a lower-bound and route the entity through the
 *   tiled path (never through the per-thread small raster).
 *
 * Returns false when no corner can be projected reliably (entirely behind
 * the camera): the bbox is meaningless and the caller should drop the
 * entity for this frame.
 *
 * NOTE: the previous implementation built a 4-vertex impostor and divided
 * x/y by clip.w with no sign check. For cylinders straddling the near plane
 * (one endpoint behind the camera, very common for cylinders), `clip.w`
 * could collapse to ~0 or flip sign, producing garbage NDC and ultimately
 * a tiny shoelace area (`quadArea`) that misclassified large cylinders as
 * "small". Those entities then drove the per-thread small raster kernel to
 * iterate the full screen, triggering WDDM preemption and the GPU context
 * churn observed in Nsight Graphics for `Cylinder: Small Raster`.
 */
__device__ inline bool computeCylinderScreenAABB(
    const glm::vec3& pa,
    const glm::vec3& pb,
    float radius,
    glm::vec2& screenMin,
    glm::vec2& screenMax,
    bool& allCornersValid)
{
    glm::vec4 camA = hybridCst.view * glm::vec4(pa, 1.0f);
    glm::vec4 camB = hybridCst.view * glm::vec4(pb, 1.0f);

    glm::vec3 axis = glm::vec3(camB) - glm::vec3(camA);
    float axisLenSq = glm::dot(axis, axis);
    glm::vec3 e = radius * glm::sqrt(glm::max(glm::vec3(0.0f),
                                              glm::vec3(1.0f) - axis * axis / fmaxf(axisLenSq, 1e-12f)));

    const glm::vec3 a = glm::vec3(camA);
    const glm::vec3 b = glm::vec3(camB);
    const glm::vec3 corners[8] = {
        a + e,
        a - e,
        b + e,
        b - e,
        a + glm::vec3( e.x, -e.y,  e.z),
        a + glm::vec3(-e.x,  e.y,  e.z),
        b + glm::vec3( e.x, -e.y,  e.z),
        b + glm::vec3(-e.x,  e.y,  e.z)
    };

    glm::vec2 minC( 1e6f);
    glm::vec2 maxC(-1e6f);
    int validCount = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        glm::vec4 clip = hybridCst.proj * glm::vec4(corners[i], 1.0f);
        if (clip.w > 1e-3f) {
            float iw = 1.0f / clip.w;
            float x = clip.x * iw;
            float y = clip.y * iw;
            minC.x = fminf(minC.x, x);
            minC.y = fminf(minC.y, y);
            maxC.x = fmaxf(maxC.x, x);
            maxC.y = fmaxf(maxC.y, y);
            ++validCount;
        }
    }

    allCornersValid = (validCount == 8);
    if (validCount == 0) {
        screenMin = glm::vec2(0.0f);
        screenMax = glm::vec2(0.0f);
        return false;
    }

    screenMin.x = floorf((minC.x * 0.5f + 0.5f) * (float)hybridCst.screenWidth );
    screenMin.y = floorf((minC.y * 0.5f + 0.5f) * (float)hybridCst.screenHeight);
    screenMax.x = ceilf ((maxC.x * 0.5f + 0.5f) * (float)hybridCst.screenWidth );
    screenMax.y = ceilf ((maxC.y * 0.5f + 0.5f) * (float)hybridCst.screenHeight);
    return true;
}

__global__ void sphereFrustumBBoxClassifyKernel(
    const glm::vec4* __restrict__ spheres,
    unsigned int* __restrict__ smallSphereIndices,
    unsigned int* __restrict__ smallSphereCount,
    unsigned long long* __restrict__ d_tile_entity_pairs,
    unsigned int* __restrict__ d_pair_count,
    unsigned int* __restrict__ frustumPassedCount,
    int maxPairs)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hybridCst.sphereCount) return;
    
    glm::vec4 spherePosR = spheres[idx];
    if (!isSphereInsideFrustum(spherePosR)) return;
    
    if (hybridCst.benchmark) atomicAdd(frustumPassedCount, 1u);
    
    glm::vec2 screenMin, screenMax;
    float centerDepth;
    computeSphereBBox(spherePosR, screenMin, screenMax, centerDepth);
    
    float width = screenMax.x - screenMin.x;
    float height = screenMax.y - screenMin.y;
    float area = width * height;
    // NOTE: Sub-pixel entities are NOT discarded here. The small-entity raster
    // uses a point-fallback (1-pixel write at the projected center) for bboxes
    // <= 1x1 to avoid concentric-ring moire artifacts caused by single-ray-
    // per-pixel sub-pixel sampling.

    if (area <= (float)hybridCst.smallEntityThreshold) 
    {
        unsigned int outIdx = atomicAdd(smallSphereCount, 1u);
        if (outIdx < (unsigned int)hybridCst.sphereCount)
            smallSphereIndices[outIdx] = idx;
    } 
    else 
    {
        int tileMinX = (int)(screenMin.x / TILE_SIZE);
        int tileMinY = (int)(screenMin.y / TILE_SIZE);
        int tileMaxX = (int)(screenMax.x / TILE_SIZE);
        int tileMaxY = (int)(screenMax.y / TILE_SIZE);
        if (tileMinX < 0) tileMinX = 0;
        if (tileMinY < 0) tileMinY = 0;
        if (tileMaxX >= hybridCst.tilesX) tileMaxX = hybridCst.tilesX - 1;
        if (tileMaxY >= hybridCst.tilesY) tileMaxY = hybridCst.tilesY - 1;
        for (int ty = tileMinY; ty <= tileMaxY; ++ty) {
            for (int tx = tileMinX; tx <= tileMaxX; ++tx) {
                unsigned int tileId = (unsigned int)(ty * hybridCst.tilesX + tx);
                unsigned long long pair = ((unsigned long long)tileId << 32) | (unsigned long long)idx;
                unsigned int outIdx = atomicAdd(d_pair_count, 1u);
                if (outIdx < (unsigned int)maxPairs)
                    d_tile_entity_pairs[outIdx] = pair;
            }
        }
    }
}

__global__ void cylinderFrustumBBoxClassifyKernel(
    const Cylinder* __restrict__ cylinders,
    const glm::vec4* __restrict__ spheres,
    unsigned int* __restrict__ smallCylinderIndices,
    unsigned int* __restrict__ smallCylinderCount,
    unsigned long long* __restrict__ d_tile_entity_pairs,
    unsigned int* __restrict__ d_pair_count,
    unsigned int* __restrict__ frustumPassedCount,
    int maxPairs)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hybridCst.cylinderCount) return;

    Cylinder cyl = cylinders[idx];
    glm::vec4 sA = spheres[cyl.sphereIndexA];
    glm::vec4 sB = spheres[cyl.sphereIndexB];
    glm::vec3 pa = glm::vec3(sA);
    glm::vec3 pb = glm::vec3(sB);
    float radius = cyl.radius;
    if (!isCylinderInsideFrustum(pa, pb, radius)) return;

    if (hybridCst.benchmark) atomicAdd(frustumPassedCount, 1u);

    glm::vec2 screenMin, screenMax;
    bool allCornersValid;
    if (!computeCylinderScreenAABB(pa, pb, radius, screenMin, screenMax, allCornersValid)) {
        // Cylinder cannot be projected reliably (entirely behind camera);
        // skip raster work to avoid feeding garbage geometry downstream.
        return;
    }

    // Clamp screen bbox to viewport before computing classification area.
    // Using the AABB (not a shoelace of an oriented quad) means the area
    // monotonically reflects the on-screen footprint and is robust against
    // degenerate impostor projections.
    float screenMinX = fmaxf(0.0f, screenMin.x);
    float screenMinY = fmaxf(0.0f, screenMin.y);
    float screenMaxX = fminf((float)hybridCst.screenWidth,  screenMax.x);
    float screenMaxY = fminf((float)hybridCst.screenHeight, screenMax.y);

    float width  = fmaxf(0.0f, screenMaxX - screenMinX);
    float height = fmaxf(0.0f, screenMaxY - screenMinY);
    float area   = width * height;
    // NOTE: Sub-pixel cylinders are NOT discarded here; the small-cylinder
    // raster uses point-fallback for tiny bboxes (see small_entity_raster.cu).

    // A cylinder straddling the near plane has a partial bbox; never let it
    // enter the per-thread small kernel (which projects an impostor that
    // would also be degenerate). Force it to the tiled path.
    if (allCornersValid && area <= (float)hybridCst.smallEntityThreshold) {
        unsigned int outIdx = atomicAdd(smallCylinderCount, 1u);
        if (outIdx < (unsigned int)hybridCst.cylinderCount)
            smallCylinderIndices[outIdx] = idx;
    }
    else
    {
        int tileMinX = (int)(screenMinX / TILE_SIZE);
        int tileMinY = (int)(screenMinY / TILE_SIZE);
        int tileMaxX = (int)(screenMaxX / TILE_SIZE);
        int tileMaxY = (int)(screenMaxY / TILE_SIZE);
        if (tileMinX < 0) tileMinX = 0;
        if (tileMinY < 0) tileMinY = 0;
        if (tileMaxX >= hybridCst.tilesX) tileMaxX = hybridCst.tilesX - 1;
        if (tileMaxY >= hybridCst.tilesY) tileMaxY = hybridCst.tilesY - 1;
        for (int ty = tileMinY; ty <= tileMaxY; ++ty) {
            for (int tx = tileMinX; tx <= tileMaxX; ++tx) {
                unsigned int tileId = (unsigned int)(ty * hybridCst.tilesX + tx);
                unsigned long long pair = ((unsigned long long)tileId << 32) | (unsigned long long)idx;
                unsigned int outIdx = atomicAdd(d_pair_count, 1u);
                if (outIdx < (unsigned int)maxPairs)
                    d_tile_entity_pairs[outIdx] = pair;
            }
        }
    }
}

extern "C" void uploadHybridConstants(const HybridConstants& constants, cudaStream_t stream) {
    cudaMemcpyToSymbolAsync(hybridCst, &constants, sizeof(HybridConstants), 0, cudaMemcpyHostToDevice, stream);
}

extern "C" void launchSphereFrustumBBoxClassify(
    const glm::vec4* d_spheres,
    unsigned int* d_smallSphereIndices,
    unsigned int* d_smallSphereCount,
    unsigned long long* d_tile_entity_pairs,
    unsigned int* d_pair_count,
    unsigned int* d_frustumPassedCount,
    int sphereCount,
    int maxPairs,
    cudaStream_t stream)
{
    dim3 block(CULL_BLOCK_SIZE);
    dim3 grid((sphereCount + block.x - 1) / block.x);
    sphereFrustumBBoxClassifyKernel<<<grid, block, 0, stream>>>(
        d_spheres, d_smallSphereIndices, d_smallSphereCount,
        d_tile_entity_pairs, d_pair_count, d_frustumPassedCount, maxPairs);
}

extern "C" void launchCylinderFrustumBBoxClassify(
    const Cylinder* d_cylinders,
    const glm::vec4* d_spheres,
    unsigned int* d_smallCylinderIndices,
    unsigned int* d_smallCylinderCount,
    unsigned long long* d_tile_entity_pairs,
    unsigned int* d_pair_count,
    unsigned int* d_frustumPassedCount,
    int cylinderCount,
    int maxPairs,
    cudaStream_t stream)
{
    dim3 block(CULL_BLOCK_SIZE);
    dim3 grid((cylinderCount + block.x - 1) / block.x);
    cylinderFrustumBBoxClassifyKernel<<<grid, block, 0, stream>>>(
        d_cylinders, d_spheres, d_smallCylinderIndices, d_smallCylinderCount,
        d_tile_entity_pairs, d_pair_count, d_frustumPassedCount, maxPairs);
}
