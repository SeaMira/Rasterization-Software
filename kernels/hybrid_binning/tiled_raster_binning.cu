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

__global__ void tiledSphereRasterBinningKernel(
    const glm::vec4* __restrict__ d_spheres,
    const unsigned int* __restrict__ d_tile_offsets,
    const unsigned long long* __restrict__ d_tile_entity_pairs_sorted,
    unsigned int* __restrict__ depthBuffer,
    cudaSurfaceObject_t outputImage)
{
    const unsigned int tileX = blockIdx.x;
    const unsigned int tileY = blockIdx.y;
    const unsigned int tileIdx = tileY * hybridCst.tilesX + tileX;
    const int pixelX = tileX * TILE_SIZE + threadIdx.x;
    const int pixelY = tileY * TILE_SIZE + threadIdx.y;
    if (pixelX >= hybridCst.screenWidth || pixelY >= hybridCst.screenHeight) return;

    const unsigned int entityStart = d_tile_offsets[tileIdx];
    const unsigned int entityEnd = d_tile_offsets[tileIdx + 1];
    const unsigned int entityCount = entityEnd - entityStart;
    if (entityCount == 0) return;

    __shared__ glm::vec4 shPosR[SHARED_BATCH];
    __shared__ int shBboxMinX[SHARED_BATCH], shBboxMinY[SHARED_BATCH];
    __shared__ int shBboxMaxX[SHARED_BATCH], shBboxMaxY[SHARED_BATCH];

    float aspectRatio = (float)hybridCst.screenWidth / (float)hybridCst.screenHeight;
    float fovRad = glm::radians(hybridCst.fov);
    float fovTan = tanf(fovRad * 0.5f);
    float halfFovTan = fovTan * aspectRatio;

    glm::vec3 rd = computeRayDirTiled(pixelX, pixelY, fovTan, halfFovTan);
    float minDepth = depthBuffer[pixelY * hybridCst.screenWidth + pixelX];
    glm::vec3 finalColor(0.0f);
    bool hasHit = false;

    unsigned int numBatches = (entityCount + SHARED_BATCH - 1) / SHARED_BATCH;
    for (unsigned int batch = 0; batch < numBatches; batch++) {
        unsigned int batchStart = batch * SHARED_BATCH;
        unsigned int localIdx = threadIdx.y * TILE_SIZE + threadIdx.x;
        if (localIdx < SHARED_BATCH) {
            unsigned int entityIdx = batchStart + localIdx;
            if (entityIdx < entityCount) {
                unsigned long long pair = d_tile_entity_pairs_sorted[entityStart + entityIdx];
                unsigned int sphereIndex = (unsigned int)(pair & 0xFFFFFFFFu);
                glm::vec4 posR = d_spheres[sphereIndex];
                shPosR[localIdx] = posR;
                int bx0, by0, bx1, by1;
                computeSphereBBoxTiled(posR, bx0, by0, bx1, by1);
                shBboxMinX[localIdx] = bx0; shBboxMinY[localIdx] = by0;
                shBboxMaxX[localIdx] = bx1; shBboxMaxY[localIdx] = by1;
            }
        }
        __syncthreads();

        unsigned int batchCount = min(SHARED_BATCH, entityCount - batchStart);
        for (unsigned int i = 0; i < batchCount; i++) {
            if (pixelX < shBboxMinX[i] || pixelX >= shBboxMaxX[i] ||
                pixelY < shBboxMinY[i] || pixelY >= shBboxMaxY[i]) continue;
            glm::vec4 posR = shPosR[i];
            float t = iSphereTiled(hybridCst.cameraPos, rd, glm::vec3(posR), posR.w);
            if (t > 0.0f) {
                glm::vec3 hit = hybridCst.cameraPos + rd * t;
                float proj22 = hybridCst.proj[2][2];
                float proj32 = hybridCst.proj[3][2];
                float depth = __fdividef(fmaf(hit.z, proj22, proj32), -hit.z);
                if (depth < minDepth) {
                    minDepth = depth;
                    hasHit = true;
                    glm::vec3 normal = glm::normalize(hit - glm::vec3(posR));
                    float lambert = fmaxf(0.0f, glm::dot(normal, -glm::normalize(rd * t)));
                    finalColor = glm::vec3(atomsColor[0], atomsColor[1], atomsColor[2]) * lambert * diffuse;
                }
            }
        }
        __syncthreads();
    }

    if (hasHit) {
        int pixelIdx = pixelY * hybridCst.screenWidth + pixelX;
        unsigned int depthU = __float_as_uint(minDepth);
        depthBuffer[pixelIdx] = depthU;
        uchar4 pixel = make_uchar4(
            (unsigned char)(finalColor.x * 255.0f),
            (unsigned char)(finalColor.y * 255.0f),
            (unsigned char)(finalColor.z * 255.0f), 255);
        surf2Dwrite(pixel, outputImage, pixelX * sizeof(uchar4), pixelY);
        
    }
}


// Cross product of 2D vectors
__device__ inline float cross2D(glm::vec2 a, glm::vec2 b) 
{
    return a.x * b.y - a.y * b.x;
}

// Check if point is inside convex quadrilateral
__device__ inline bool pointInQuad(glm::vec2 p, glm::vec2 q0, glm::vec2 q1, glm::vec2 q2, glm::vec2 q3) 
{
    float c0 = cross2D(q1 - q0, p - q0);
    float c1 = cross2D(q2 - q1, p - q1);
    float c2 = cross2D(q3 - q2, p - q2);
    float c3 = cross2D(q0 - q3, p - q3);
    
    return (c0 >= 0 && c1 >= 0 && c2 >= 0 && c3 >= 0) ||
           (c0 <= 0 && c1 <= 0 && c2 <= 0 && c3 <= 0);
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

__global__ void tiledCylinderRasterBinningKernel(
    const Cylinder* __restrict__ d_cylinders,
    const unsigned int* __restrict__ d_tile_offsets,
    const unsigned long long* __restrict__ d_tile_entity_pairs_sorted,
    unsigned int* __restrict__ depthBuffer,
    cudaSurfaceObject_t outputImage)
{
    const unsigned int tileX = blockIdx.x;
    const unsigned int tileY = blockIdx.y;
    const unsigned int tileIdx = tileY * hybridCst.tilesX + tileX;
    const int pixelX = tileX * TILE_SIZE + threadIdx.x;
    const int pixelY = tileY * TILE_SIZE + threadIdx.y;
    if (pixelX >= hybridCst.screenWidth || pixelY >= hybridCst.screenHeight) return;

    const unsigned int entityStart = d_tile_offsets[tileIdx];
    const unsigned int entityEnd = d_tile_offsets[tileIdx + 1];
    const unsigned int entityCount = entityEnd - entityStart;
    if (entityCount == 0) return;

    __shared__ glm::vec4 shPa[SHARED_BATCH], shPb[SHARED_BATCH];
    __shared__ glm::vec2 shQuad[SHARED_BATCH * 4];

    float aspectRatio = (float)hybridCst.screenWidth / (float)hybridCst.screenHeight;
    float fovRad = glm::radians(hybridCst.fov);
    float fovTan = tanf(fovRad * 0.5f);
    float halfFovTan = fovTan * aspectRatio;
    glm::vec3 rd = computeRayDirTiled(pixelX, pixelY, fovTan, halfFovTan);
    float minDepth = depthBuffer[pixelY * hybridCst.screenWidth + pixelX];
    glm::vec3 finalColor(0.0f);
    bool hasHit = false;

    unsigned int numBatches = (entityCount + SHARED_BATCH - 1) / SHARED_BATCH;
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

                glm::vec3 pa = glm::vec3(cyl.pa_r), pb = glm::vec3(cyl.pb_r);
                float radius = cyl.pa_r.w;
                glm::vec4 camA = hybridCst.view * glm::vec4(pa, 1.0f);
                glm::vec4 camB = hybridCst.view * glm::vec4(pb, 1.0f);
                glm::vec3 camImpPosA, camImpPosB;
                if ( camA.z < camB.z )
                {
                    camImpPosA = glm::vec3(camB);
                    camImpPosB = glm::vec3(camA);
                }
                else
                {
                    camImpPosA = glm::vec3(camA);
                    camImpPosB = glm::vec3(camB);
                }
                glm::vec3 center = glm::normalize( ( camImpPosA + camImpPosB ) * 0.5f );
                // Cylinder axis
                const glm::vec3 z = glm::normalize(camImpPosB - camImpPosA);

                // Find orthonormal x,y axes orthogonal to cylinder axis
                glm::vec3 x = glm::normalize(glm::cross(center, z));
                glm::vec3 y = glm::normalize(glm::cross(x, z)); // make full basis

                // Compute impostor construction vectors.
                const float dV0 = glm::length( camImpPosA );
                const float dV1 = glm::length( camImpPosB );

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

                shQuad[localIdx * 4 + 0] = glm::vec2(ndcv1Proj);
                shQuad[localIdx * 4 + 1] = glm::vec2(ndcv2Proj);
                shQuad[localIdx * 4 + 2] = glm::vec2(ndcv4Proj);
                shQuad[localIdx * 4 + 3] = glm::vec2(ndcv3Proj);
            }
        }
        __syncthreads();

        unsigned int batchCount = min(SHARED_BATCH, entityCount - batchStart);
        glm::vec2 pixelCenter((float)pixelX + 0.5f, (float)pixelY + 0.5f);
        for (unsigned int i = 0; i < batchCount; i++) {
            glm::vec2 q0 = shQuad[i * 4 + 0], q1 = shQuad[i * 4 + 1];
            glm::vec2 q2 = shQuad[i * 4 + 2], q3 = shQuad[i * 4 + 3];
            
            if (!pointInQuad(pixelCenter, q0, q1, q2, q3)) continue;
            glm::vec3 pa = glm::vec3(shPa[i]), pb = glm::vec3(shPb[i]);
            float ra = shPa[i].w;
            glm::vec4 tnor = iCylinderTiled(hybridCst.cameraPos, rd, pa, pb, ra);
            if (tnor.x > 0.0f) {
                float t = tnor.x;
                glm::vec3 hit = hybridCst.cameraPos + rd * t;
                float depth = __fdividef(fmaf(hit.z, hybridCst.proj[2][2], hybridCst.proj[3][2]), -hit.z);

                if (depth < minDepth) {
                    minDepth = depth;
                    hasHit = true;
                    glm::vec3 normal = glm::normalize(glm::vec3(tnor.y, tnor.z, tnor.w));
                    float lambert = glm::max(0.0f, glm::dot(normal, -glm::normalize(rd * t)));
                    finalColor = glm::vec3(bondsColor[0], bondsColor[1], bondsColor[2]) * lambert * diffuse;
                }
            }
        }
        __syncthreads();
    }

    if (hasHit) {
        int pixelIdx = pixelY * hybridCst.screenWidth + pixelX;
        unsigned int depthU = __float_as_uint(minDepth);
        depthBuffer[pixelIdx] = depthU;
        uchar4 pixel = make_uchar4(
            (unsigned char)(finalColor.x * 255.0f),
            (unsigned char)(finalColor.y * 255.0f),
            (unsigned char)(finalColor.z * 255.0f), 255);
        surf2Dwrite(pixel, outputImage, pixelX * sizeof(uchar4), pixelY);
    }
}

extern "C" void launchTiledSphereRasterBinning(
    const glm::vec4* d_spheres,
    const unsigned int* d_tile_offsets,
    const unsigned long long* d_tile_entity_pairs_sorted,
    unsigned int* d_depthBuffer,
    cudaSurfaceObject_t outputImage,
    unsigned int tilesX,
    unsigned int tilesY,
    cudaStream_t stream)
{
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(tilesX, tilesY);
    tiledSphereRasterBinningKernel<<<grid, block, 0, stream>>>(
        d_spheres, d_tile_offsets, d_tile_entity_pairs_sorted, d_depthBuffer, outputImage);
}

extern "C" void launchTiledCylinderRasterBinning(
    const Cylinder* d_cylinders,
    const unsigned int* d_tile_offsets,
    const unsigned long long* d_tile_entity_pairs_sorted,
    unsigned int* d_depthBuffer,
    cudaSurfaceObject_t outputImage,
    unsigned int tilesX,
    unsigned int tilesY,
    cudaStream_t stream)
{
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(tilesX, tilesY);
    tiledCylinderRasterBinningKernel<<<grid, block, 0, stream>>>(
        d_cylinders, d_tile_offsets, d_tile_entity_pairs_sorted, d_depthBuffer, outputImage);
}
