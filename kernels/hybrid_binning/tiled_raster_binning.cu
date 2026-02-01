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
    int& bboxMinX, int& bboxMinY, int& bboxMaxX, int& bboxMaxY) {
    glm::vec4 camSpace4 = hybridCst.view * glm::vec4(glm::vec3(spherePosR), 1.0f);
    glm::vec3 cameraSpaceSphere = glm::vec3(camSpace4);
    glm::vec3 normCamSpaceSphere = glm::normalize(cameraSpaceSphere);
    glm::vec3 camImposPos = cameraSpaceSphere - normCamSpaceSphere * spherePosR.w;
    float dist = glm::length(cameraSpaceSphere) + 1e-6f;
    float sinAngle = spherePosR.w / dist;
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
    float minCx = 1e6f, minCy = 1e6f, maxCx = -1e6f, maxCy = -1e6f;
    for (int i = 0; i < 4; i++) {
        glm::vec4 clip = hybridCst.proj * glm::vec4(corners[i], 1.0f);
        float iw = 1.0f / clip.w;
        float x = clip.x * iw, y = clip.y * iw;
        minCx = fminf(minCx, x); minCy = fminf(minCy, y);
        maxCx = fmaxf(maxCx, x); maxCy = fmaxf(maxCy, y);
    }
    bboxMinX = (int)floorf((minCx * 0.5f + 0.5f) * hybridCst.screenWidth);
    bboxMinY = (int)floorf((minCy * 0.5f + 0.5f) * hybridCst.screenHeight);
    bboxMaxX = (int)ceilf((maxCx * 0.5f + 0.5f) * hybridCst.screenWidth);
    bboxMaxY = (int)ceilf((maxCy * 0.5f + 0.5f) * hybridCst.screenHeight);
    bboxMinX = max(0, bboxMinX); bboxMinY = max(0, bboxMinY);
    bboxMaxX = min(hybridCst.screenWidth, bboxMaxX);
    bboxMaxY = min(hybridCst.screenHeight, bboxMaxY);
}

__global__ void tiledSphereRasterBinningKernel(
    const glm::vec4* __restrict__ d_spheres,
    const unsigned int* __restrict__ d_tile_offsets,
    const unsigned int* __restrict__ d_sorted_entity_indices,
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
    float minDepth = 1e10f;
    glm::vec3 finalColor(0.0f);
    bool hasHit = false;

    unsigned int numBatches = (entityCount + SHARED_BATCH - 1) / SHARED_BATCH;
    for (unsigned int batch = 0; batch < numBatches; batch++) {
        unsigned int batchStart = batch * SHARED_BATCH;
        unsigned int localIdx = threadIdx.y * TILE_SIZE + threadIdx.x;
        if (localIdx < SHARED_BATCH) {
            unsigned int entityIdx = batchStart + localIdx;
            if (entityIdx < entityCount) {
                unsigned int sphereIndex = d_sorted_entity_indices[entityStart + entityIdx];
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
                glm::vec4 hitClip = hybridCst.proj * (hybridCst.view * glm::vec4(hit, 1.0f));
                float depth = hitClip.z / hitClip.w;
                if (depth < minDepth) {
                    minDepth = depth;
                    hasHit = true;
                    glm::vec3 normal = glm::normalize(hit - glm::vec3(posR));
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
                (unsigned char)(finalColor.z * 255.0f), 255);
            surf2Dwrite(pixel, outputImage, pixelX * sizeof(uchar4), pixelY);
        }
    }
}

// Cylinder: ray-cylinder and bbox from quad (simplified - reuse logic from hybrid tiled)
__device__ inline glm::vec4 iCylinderTiled(const glm::vec3& ro, const glm::vec3& rd,
    const glm::vec3& pa, const glm::vec3& pb, float ra) {
    glm::vec3 ba = pb - pa, oc = ro - pa;
    float baba = glm::dot(ba, ba), bard = glm::dot(ba, rd), baoc = glm::dot(ba, oc);
    float k2 = baba - bard * bard;
    float k1 = baba * glm::dot(oc, rd) - baoc * bard;
    float k0 = baba * glm::dot(oc, oc) - baoc * baoc - ra * ra * baba;
    float h = k1 * k1 - k2 * k0;
    if (h < 0.0f) return glm::vec4(-1.0f);
    h = sqrtf(h);
    float t = (-k1 - h) / k2;
    float y = baoc + t * bard;
    if (y > 0.0f && y < baba) return glm::vec4(t, oc + t * rd - ba * y / baba);
    return glm::vec4(-1.0f);
}

__global__ void tiledCylinderRasterBinningKernel(
    const Cylinder* __restrict__ d_cylinders,
    const unsigned int* __restrict__ d_tile_offsets,
    const unsigned int* __restrict__ d_sorted_entity_indices,
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
    float minDepth = 1e10f;
    glm::vec3 finalColor(0.0f);
    bool hasHit = false;

    unsigned int numBatches = (entityCount + SHARED_BATCH - 1) / SHARED_BATCH;
    for (unsigned int batch = 0; batch < numBatches; batch++) {
        unsigned int batchStart = batch * SHARED_BATCH;
        unsigned int localIdx = threadIdx.y * TILE_SIZE + threadIdx.x;
        if (localIdx < SHARED_BATCH) {
            unsigned int entityIdx = batchStart + localIdx;
            if (entityIdx < entityCount) {
                unsigned int cylIndex = d_sorted_entity_indices[entityStart + entityIdx];
                Cylinder cyl = d_cylinders[cylIndex];
                shPa[localIdx] = cyl.pa_r;
                shPb[localIdx] = cyl.pb_r;
                glm::vec3 pa = glm::vec3(cyl.pa_r), pb = glm::vec3(cyl.pb_r);
                float radius = cyl.pa_r.w;
                glm::vec4 camA = hybridCst.view * glm::vec4(pa, 1.0f);
                glm::vec4 camB = hybridCst.view * glm::vec4(pb, 1.0f);
                glm::vec3 camImpPosA = glm::vec3(camA), camImpPosB = glm::vec3(camB);
                if (camImpPosA.z > camImpPosB.z) { glm::vec3 tmp = camImpPosA; camImpPosA = camImpPosB; camImpPosB = tmp; }
                glm::vec3 z = glm::normalize(camImpPosB - camImpPosA);
                glm::vec3 center = glm::normalize((camImpPosA + camImpPosB) * 0.5f);
                glm::vec3 x = glm::normalize(glm::cross(center, z));
                glm::vec3 y = glm::normalize(glm::cross(x, z));
                float dV0 = glm::length(camImpPosA), dV1 = glm::length(camImpPosB);
                float sinA0 = radius / dV0, sinA1 = radius / dV1;
                float angle0 = asinf(fminf(sinA0, 0.999f)), angle1 = asinf(fminf(sinA1, 0.999f));
                glm::vec3 v1 = camImpPosA - x * (radius * cosf(angle0)) + y * (radius * sinA0);
                glm::vec3 v2 = camImpPosA + x * (radius * cosf(angle0)) + y * (radius * sinA0);
                glm::vec3 v3 = camImpPosB - x * (float)(dV1 - radius) * tanf(angle1) + y * radius;
                glm::vec3 v4 = camImpPosB + x * (float)(dV1 - radius) * tanf(angle1) + y * radius;
                for (int k = 0; k < 4; k++) {
                    glm::vec3 v = (k == 0) ? v1 : (k == 1) ? v2 : (k == 2) ? v4 : v3;
                    glm::vec4 clip = hybridCst.proj * glm::vec4(v, 1.0f);
                    glm::vec2 ndc = glm::vec2(clip.x / clip.w, clip.y / clip.w);
                    shQuad[localIdx * 4 + k] = (ndc * 0.5f + 0.5f) * glm::vec2(hybridCst.screenWidth, hybridCst.screenHeight);
                }
            }
        }
        __syncthreads();

        unsigned int batchCount = min(SHARED_BATCH, entityCount - batchStart);
        glm::vec2 pixelCenter((float)pixelX + 0.5f, (float)pixelY + 0.5f);
        for (unsigned int i = 0; i < batchCount; i++) {
            glm::vec2 q0 = shQuad[i * 4 + 0], q1 = shQuad[i * 4 + 1];
            glm::vec2 q2 = shQuad[i * 4 + 2], q3 = shQuad[i * 4 + 3];
            float c0 = (q1.x - q0.x) * (pixelCenter.y - q0.y) - (q1.y - q0.y) * (pixelCenter.x - q0.x);
            float c1 = (q2.x - q1.x) * (pixelCenter.y - q1.y) - (q2.y - q1.y) * (pixelCenter.x - q1.x);
            float c2 = (q3.x - q2.x) * (pixelCenter.y - q2.y) - (q3.y - q2.y) * (pixelCenter.x - q2.x);
            float c3 = (q0.x - q3.x) * (pixelCenter.y - q3.y) - (q0.y - q3.y) * (pixelCenter.x - q3.x);
            bool inside = (c0 >= 0 && c1 >= 0 && c2 >= 0 && c3 >= 0) || (c0 <= 0 && c1 <= 0 && c2 <= 0 && c3 <= 0);
            if (!inside) continue;
            glm::vec3 pa = glm::vec3(shPa[i]), pb = glm::vec3(shPb[i]);
            float ra = shPa[i].w;
            glm::vec4 tnor = iCylinderTiled(hybridCst.cameraPos, rd, pa, pb, ra);
            if (tnor.x > 0.0f) {
                glm::vec3 hit = hybridCst.cameraPos + rd * tnor.x;
                glm::vec4 hitClip = hybridCst.proj * (hybridCst.view * glm::vec4(hit, 1.0f));
                float depth = hitClip.z / hitClip.w;
                if (depth < minDepth) {
                    minDepth = depth;
                    hasHit = true;
                    glm::vec3 normal = glm::normalize(tnor.yzw);
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
                (unsigned char)(finalColor.z * 255.0f), 255);
            surf2Dwrite(pixel, outputImage, pixelX * sizeof(uchar4), pixelY);
        }
    }
}

extern "C" void launchTiledSphereRasterBinning(
    const glm::vec4* d_spheres,
    const unsigned int* d_tile_offsets,
    const unsigned int* d_sorted_entity_indices,
    unsigned int* d_depthBuffer,
    cudaSurfaceObject_t outputImage,
    unsigned int tilesX,
    unsigned int tilesY,
    cudaStream_t stream)
{
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(tilesX, tilesY);
    tiledSphereRasterBinningKernel<<<grid, block, 0, stream>>>(
        d_spheres, d_tile_offsets, d_sorted_entity_indices, d_depthBuffer, outputImage);
}

extern "C" void launchTiledCylinderRasterBinning(
    const Cylinder* d_cylinders,
    const unsigned int* d_tile_offsets,
    const unsigned int* d_sorted_entity_indices,
    unsigned int* d_depthBuffer,
    cudaSurfaceObject_t outputImage,
    unsigned int tilesX,
    unsigned int tilesY,
    cudaStream_t stream)
{
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(tilesX, tilesY);
    tiledCylinderRasterBinningKernel<<<grid, block, 0, stream>>>(
        d_cylinders, d_tile_offsets, d_sorted_entity_indices, d_depthBuffer, outputImage);
}
