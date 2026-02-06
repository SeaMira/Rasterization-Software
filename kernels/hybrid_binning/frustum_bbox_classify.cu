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

__device__ inline void computeCylinderBBox(
    const glm::vec3& pa,
    const glm::vec3& pb,
    float radius,
    glm::vec2 projectedPoints[4])
{
    glm::vec4 camA = hybridCst.view * glm::vec4(pa, 1.0f);
    glm::vec4 camB = hybridCst.view * glm::vec4(pb, 1.0f);
    
    glm::vec3 camImpPosA, camImpPosB;
    if ( camA.z < camB.z )
	{
		camImpPosA = camB;
		camImpPosB = camA;
	}
	else
	{
		camImpPosA = camA;
		camImpPosB = camB;
	}
    glm::vec3 center = normalize( ( camImpPosA + camImpPosB ) * 0.5f );
    // Cylinder axis
    const glm::vec3 z = normalize(camImpPosB - camImpPosA);

    // Find orthonormal x,y axes orthogonal to cylinder axis
    glm::vec3 x = normalize(cross(center, z));
    glm::vec3 y = normalize(cross(x, z)); // make full basis

    // Compute impostor construction vectors.
    const float dV0 = length( camImpPosA );
    const float dV1 = length( camImpPosB );

    const float sinAngle = __fdividef(radius, dV0);
    float		angle	 = asinf( sinAngle );
    const glm::vec3	y1		 = y * radius;
    const glm::vec3	x2		 = x * radius * cosf( angle );
    const glm::vec3	y2		 = y1 * sinAngle;
    angle				 = asinf( __fdividef(radius, dV1) );
    const glm::vec3 x3		 = x * ( dV1 - radius ) * __tanf( angle );

    // Compute impostors vertices.
    const glm::vec3 v1 = camImpPosA - x2 + y2;
    const glm::vec3 v2 = camImpPosA + x2 + y2;
    const glm::vec3 v3 = camImpPosB - x3 + y1;
    const glm::vec3 v4 = camImpPosB + x3 + y1;

    const glm::vec4 v1Proj = hybridCst.proj * glm::vec4(v1, 1.0f);
    const glm::vec4 v2Proj = hybridCst.proj * glm::vec4(v2, 1.0f);
    const glm::vec4 v3Proj = hybridCst.proj * glm::vec4(v3, 1.0f);
    const glm::vec4 v4Proj = hybridCst.proj * glm::vec4(v4, 1.0f);

    glm::vec3 ndcv1Proj = glm::vec3(__fdividef(v1Proj.x, v1Proj.w), __fdividef(v1Proj.y, v1Proj.w), __fdividef(v1Proj.z, v1Proj.w));
    glm::vec3 ndcv2Proj = glm::vec3(__fdividef(v2Proj.x, v2Proj.w), __fdividef(v2Proj.y, v2Proj.w), __fdividef(v2Proj.z, v2Proj.w));
    glm::vec3 ndcv3Proj = glm::vec3(__fdividef(v3Proj.x, v3Proj.w), __fdividef(v3Proj.y, v3Proj.w), __fdividef(v3Proj.z, v3Proj.w));
    glm::vec3 ndcv4Proj = glm::vec3(__fdividef(v4Proj.x, v4Proj.w), __fdividef(v4Proj.y, v4Proj.w), __fdividef(v4Proj.z, v4Proj.w));

    projectedPoints[0] = glm::vec2(ndcv1Proj);
    projectedPoints[1] = glm::vec2(ndcv2Proj);
    projectedPoints[2] = glm::vec2(ndcv4Proj);
    projectedPoints[3] = glm::vec2(ndcv3Proj);
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
    if (area < 1.0f) return;
    
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
        int tileMaxY = (int)(screenMax.y / TILE_SIZE)+1;
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

// Compute area of quadrilateral using shoelace formula
__device__ inline float quadArea(const glm::vec2& q0, const glm::vec2& q1, const glm::vec2& q2, const glm::vec2& q3) 
{
    return 0.5f * fabsf(
        (q0.x * q1.y - q1.x * q0.y) +
        (q1.x * q2.y - q2.x * q1.y) +
        (q2.x * q3.y - q3.x * q2.y) +
        (q3.x * q0.y - q0.x * q3.y)
    );
}

__global__ void cylinderFrustumBBoxClassifyKernel(
    const Cylinder* __restrict__ cylinders,
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
    glm::vec3 pa = glm::vec3(cyl.pa_r);
    glm::vec3 pb = glm::vec3(cyl.pb_r);
    float radius = cyl.pa_r.w;
    if (!isCylinderInsideFrustum(pa, pb, radius)) return;
    
    if (hybridCst.benchmark) atomicAdd(frustumPassedCount, 1u);
    
    glm::vec2 projectedPoints[4];
    computeCylinderBBox(pa, pb, radius, projectedPoints);

    // Convert NDC to screen coordinates and compute screen bbox
    float screenMinX = 1e6f, screenMinY = 1e6f, screenMaxX = -1e6f, screenMaxY = -1e6f;
    for (int i = 0; i < 4; i++) 
    {
        float sx = fmaf(projectedPoints[i].x, 0.5f, 0.5f) * (float)hybridCst.screenWidth;
        float sy = fmaf(projectedPoints[i].y, 0.5f, 0.5f) * (float)hybridCst.screenHeight;
        if (sx < screenMinX) screenMinX = sx;
        if (sy < screenMinY) screenMinY = sy;
        if (sx > screenMaxX) screenMaxX = sx;
        if (sy > screenMaxY) screenMaxY = sy;
    }
    screenMinX = fmaxf(0.0f, screenMinX);
    screenMinY = fmaxf(0.0f, screenMinY);
    screenMaxX = fminf((float)hybridCst.screenWidth, screenMaxX);
    screenMaxY = fminf((float)hybridCst.screenHeight, screenMaxY);
    
    float area = quadArea(projectedPoints[0], projectedPoints[1], projectedPoints[2], projectedPoints[3]);
    if (area < 1.0f) return;
    
    if (area <= (float)hybridCst.smallEntityThreshold) {
        unsigned int outIdx = atomicAdd(smallCylinderCount, 1u);
        if (outIdx < (unsigned int)hybridCst.cylinderCount)
            smallCylinderIndices[outIdx] = idx;
    } else {
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
        d_cylinders, d_smallCylinderIndices, d_smallCylinderCount,
        d_tile_entity_pairs, d_pair_count, d_frustumPassedCount, maxPairs);
}
