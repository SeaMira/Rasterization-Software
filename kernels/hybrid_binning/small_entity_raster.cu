/**
 * @file small_entity_raster.cu
 * @brief Direct rasterization for small entities (1 thread per entity). Same as hybrid.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>

#include "hybrid_binning_types.h"
#include "geometry/cylinder/cylinder.h"

extern __constant__ HybridConstants hybridCst;

__device__ inline float iSphere(const glm::vec3& ro, const glm::vec3& rd,
                                const glm::vec3& center, float radius) {
    glm::vec3 oc = ro - center;
    float b = glm::dot(oc, rd);
    float c = glm::dot(oc, oc) - radius * radius;
    float h = b * b - c;
    if (h < 0.0f) return -1.0f;
    return -b - sqrtf(h);
}

__device__ inline glm::vec2 iCylinder(const glm::vec3& ro, const glm::vec3& rd,
                                      const glm::vec3& pa, const glm::vec3& pb, float ra) {
    glm::vec3 ba = pb - pa, oc = ro - pa;
    float baba = glm::dot(ba, ba), bard = glm::dot(ba, rd), baoc = glm::dot(ba, oc);
    float k2 = baba - bard * bard;
    float k1 = baba * glm::dot(oc, rd) - baoc * bard;
    float k0 = baba * glm::dot(oc, oc) - baoc * baoc - ra * ra * baba;
    float h = k1 * k1 - k2 * k0;
    if (h < 0.0f) return glm::vec2(-1.0f);
    h = sqrtf(h);
    float t = (-k1 - h) / k2;
    float y = baoc + t * bard;
    if (y > 0.0f && y < baba) return glm::vec2(t, y / baba);
    t = ((y < 0.0f ? 0.0f : baba) - baoc) / bard;
    if (fabsf(k1 + k2 * t) < h) return glm::vec2(t, y < 0.0f ? 0.0f : 1.0f);
    return glm::vec2(-1.0f);
}

__device__ inline glm::vec3 computeRayDirection(int px, int py, float fovTan, float halfFovTan) {
    glm::vec2 p = (-glm::vec2(hybridCst.screenWidth, hybridCst.screenHeight) +
                   2.0f * glm::vec2(px, py)) / glm::vec2(hybridCst.screenWidth, hybridCst.screenHeight);
    return glm::normalize(p.x * hybridCst.right * halfFovTan + p.y * hybridCst.up * fovTan + hybridCst.front);
}

__global__ void smallSphereRasterKernel(
    const glm::vec4* __restrict__ spheres,
    const unsigned int* __restrict__ smallSphereIndices,
    unsigned int smallSphereCount,
    unsigned int* __restrict__ depthBuffer,
    cudaSurfaceObject_t outputImage)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= smallSphereCount) return;
    unsigned int sphereIdx = smallSphereIndices[idx];
    glm::vec4 spherePosR = spheres[sphereIdx];
    glm::vec3 spherePos = glm::vec3(spherePosR);
    float radius = spherePosR.w;
    glm::vec4 camSpace4 = hybridCst.view * glm::vec4(spherePos, 1.0f);
    glm::vec3 cameraSpaceSphere = glm::vec3(camSpace4);
    glm::vec3 normCamSpaceSphere = glm::normalize(cameraSpaceSphere);
    glm::vec3 camImposPos = cameraSpaceSphere - normCamSpaceSphere * radius;
    float dist = glm::length(cameraSpaceSphere) + 1e-6f;
    float sinAngle = radius / dist;
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
    for (int i = 0; i < 4; i++) {
        glm::vec4 clip = hybridCst.proj * glm::vec4(corners[i], 1.0f);
        float iw = 1.0f / clip.w;
        minC.x = fminf(minC.x, clip.x * iw); minC.y = fminf(minC.y, clip.y * iw);
        maxC.x = fmaxf(maxC.x, clip.x * iw); maxC.y = fmaxf(maxC.y, clip.y * iw);
    }
    int screenMinX = max(0, (int)floorf((minC.x * 0.5f + 0.5f) * hybridCst.screenWidth));
    int screenMinY = max(0, (int)floorf((minC.y * 0.5f + 0.5f) * hybridCst.screenHeight));
    int screenMaxX = min(hybridCst.screenWidth, (int)ceilf((maxC.x * 0.5f + 0.5f) * hybridCst.screenWidth));
    int screenMaxY = min(hybridCst.screenHeight, (int)ceilf((maxC.y * 0.5f + 0.5f) * hybridCst.screenHeight));
    float aspectRatio = (float)hybridCst.screenWidth / (float)hybridCst.screenHeight;
    float fovRad = glm::radians(hybridCst.fov);
    float fovTan = tanf(fovRad * 0.5f);
    float halfFovTan = fovTan * aspectRatio;
    for (int py = screenMinY; py < screenMaxY; py++) {
        for (int px = screenMinX; px < screenMaxX; px++) {
            glm::vec3 rd = computeRayDirection(px, py, fovTan, halfFovTan);
            float t = iSphere(hybridCst.cameraPos, rd, spherePos, radius);
            if (t > 0.0f) {
                glm::vec3 hit = hybridCst.cameraPos + rd * t;
                glm::vec4 hitClip = hybridCst.proj * (hybridCst.view * glm::vec4(hit, 1.0f));
                float depth = hitClip.z / hitClip.w;
                unsigned int depthU = __float_as_uint(depth);
                int pixelIdx = py * hybridCst.screenWidth + px;
                if (atomicMin(&depthBuffer[pixelIdx], depthU) != depthU) {
                    glm::vec3 normal = glm::normalize(hit - spherePos);
                    float lambert = fmaxf(0.0f, glm::dot(normal, -glm::normalize(rd)));
                    glm::vec3 color = glm::vec3(0.01f, 1.0f, 0.05f) * lambert * 0.9f;
                    uchar4 pixel = make_uchar4((unsigned char)(color.x * 255.0f), (unsigned char)(color.y * 255.0f), (unsigned char)(color.z * 255.0f), 255);
                    surf2Dwrite(pixel, outputImage, px * sizeof(uchar4), py);
                }
            }
        }
    }
}

__global__ void smallCylinderRasterKernel(
    const Cylinder* __restrict__ cylinders,
    const unsigned int* __restrict__ smallCylinderIndices,
    unsigned int smallCylinderCount,
    unsigned int* __restrict__ depthBuffer,
    cudaSurfaceObject_t outputImage)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= smallCylinderCount) return;
    unsigned int cylIdx = smallCylinderIndices[idx];
    Cylinder cyl = cylinders[cylIdx];
    glm::vec3 pa = glm::vec3(cyl.pa_r), pb = glm::vec3(cyl.pb_r);
    float radius = cyl.pa_r.w;
    glm::vec4 camA = hybridCst.view * glm::vec4(pa, 1.0f);
    glm::vec4 camB = hybridCst.view * glm::vec4(pb, 1.0f);
    glm::vec3 axis = glm::vec3(camB) - glm::vec3(camA);
    float axisLenSq = glm::dot(axis, axis);
    glm::vec3 e = radius * glm::sqrt(glm::max(glm::vec3(0.001f), glm::vec3(1.0f) - axis * axis / axisLenSq));
    glm::vec3 pts[4] = { glm::vec3(camA) + e, glm::vec3(camA) - e, glm::vec3(camB) + e, glm::vec3(camB) - e };
    glm::vec2 minC(1e6f), maxC(-1e6f);
    for (int i = 0; i < 4; i++) {
        glm::vec4 clip = hybridCst.proj * glm::vec4(pts[i], 1.0f);
        if (clip.w > 0.001f) {
            float iw = 1.0f / clip.w;
            minC.x = fminf(minC.x, clip.x * iw); minC.y = fminf(minC.y, clip.y * iw);
            maxC.x = fmaxf(maxC.x, clip.x * iw); maxC.y = fmaxf(maxC.y, clip.y * iw);
        }
    }
    int screenMinX = max(0, (int)floorf((minC.x * 0.5f + 0.5f) * hybridCst.screenWidth));
    int screenMinY = max(0, (int)floorf((minC.y * 0.5f + 0.5f) * hybridCst.screenHeight));
    int screenMaxX = min(hybridCst.screenWidth, (int)ceilf((maxC.x * 0.5f + 0.5f) * hybridCst.screenWidth));
    int screenMaxY = min(hybridCst.screenHeight, (int)ceilf((maxC.y * 0.5f + 0.5f) * hybridCst.screenHeight));
    float aspectRatio = (float)hybridCst.screenWidth / (float)hybridCst.screenHeight;
    float fovRad = glm::radians(hybridCst.fov);
    float fovTan = tanf(fovRad * 0.5f);
    float halfFovTan = fovTan * aspectRatio;
    for (int py = screenMinY; py < screenMaxY; py++) {
        for (int px = screenMinX; px < screenMaxX; px++) {
            glm::vec3 rd = computeRayDirection(px, py, fovTan, halfFovTan);
            glm::vec2 result = iCylinder(hybridCst.cameraPos, rd, pa, pb, radius);
            if (result.x > 0.0f) {
                glm::vec3 hit = hybridCst.cameraPos + rd * result.x;
                glm::vec4 hitClip = hybridCst.proj * (hybridCst.view * glm::vec4(hit, 1.0f));
                float depth = hitClip.z / hitClip.w;
                unsigned int depthU = __float_as_uint(depth);
                int pixelIdx = py * hybridCst.screenWidth + px;
                if (atomicMin(&depthBuffer[pixelIdx], depthU) != depthU) {
                    glm::vec3 ba = pb - pa, hitLocal = hit - pa;
                    float h = glm::dot(hitLocal, ba) / glm::dot(ba, ba);
                    glm::vec3 normal = glm::normalize(hitLocal - ba * glm::clamp(h, 0.0f, 1.0f));
                    float lambert = fmaxf(0.0f, glm::dot(normal, -glm::normalize(rd)));
                    glm::vec3 color = glm::vec3(0.01f, 1.0f, 0.05f) * lambert * 0.9f;
                    uchar4 pixel = make_uchar4((unsigned char)(color.x * 255.0f), (unsigned char)(color.y * 255.0f), (unsigned char)(color.z * 255.0f), 255);
                    surf2Dwrite(pixel, outputImage, px * sizeof(uchar4), py);
                }
            }
        }
    }
}

extern "C" void launchSmallSphereRaster(
    const glm::vec4* d_spheres,
    const unsigned int* d_smallSphereIndices,
    unsigned int smallSphereCount,
    unsigned int* d_depthBuffer,
    cudaSurfaceObject_t outputImage,
    cudaStream_t stream)
{
    if (smallSphereCount == 0) return;
    dim3 block(256);
    dim3 grid((smallSphereCount + block.x - 1) / block.x);
    smallSphereRasterKernel<<<grid, block, 0, stream>>>(d_spheres, d_smallSphereIndices, smallSphereCount, d_depthBuffer, outputImage);
}

extern "C" void launchSmallCylinderRaster(
    const Cylinder* d_cylinders,
    const unsigned int* d_smallCylinderIndices,
    unsigned int smallCylinderCount,
    unsigned int* d_depthBuffer,
    cudaSurfaceObject_t outputImage,
    cudaStream_t stream)
{
    if (smallCylinderCount == 0) return;
    dim3 block(256);
    dim3 grid((smallCylinderCount + block.x - 1) / block.x);
    smallCylinderRasterKernel<<<grid, block, 0, stream>>>(d_cylinders, d_smallCylinderIndices, smallCylinderCount, d_depthBuffer, outputImage);
}
