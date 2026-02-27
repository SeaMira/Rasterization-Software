/**
 * @file sphere_raster_ooc.cu
 * @brief Direct sphere rasterisation for the out-of-core pipeline.
 *
 * One thread per atom: projects a screen-space bounding box, then
 * ray-tests each pixel for exact sphere intersection.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>

#include "outofcore/outofcore_types.h"

static __constant__ OocConstants oocRasterCst;

extern "C" void uploadOocRasterConstants(const OocConstants& constants, cudaStream_t stream) {
    cudaMemcpyToSymbolAsync(oocRasterCst, &constants, sizeof(OocConstants), 0,
                            cudaMemcpyHostToDevice, stream);
}

__device__ inline float iSphereOoc(const glm::vec3& ro, const glm::vec3& rd,
                                   const glm::vec3& center, float radius) {
    glm::vec3 oc = ro - center;
    float b = glm::dot(oc, rd);
    float c = glm::dot(oc, oc) - radius * radius;
    float h = b * b - c;
    if (h < 0.0f) return -1.0f;
    return -b - sqrtf(h);
}

__device__ inline glm::vec3 computeRayOoc(int px, int py) {
    float aspect = static_cast<float>(oocRasterCst.screenWidth) /
                   static_cast<float>(oocRasterCst.screenHeight);
    float fovRad = glm::radians(oocRasterCst.fov);
    float fovTan = tanf(fovRad * 0.5f);
    float halfFovTan = fovTan * aspect;

    glm::vec2 p = (-glm::vec2(oocRasterCst.screenWidth, oocRasterCst.screenHeight) +
                   2.0f * glm::vec2(px, py)) /
                  glm::vec2(oocRasterCst.screenWidth, oocRasterCst.screenHeight);

    glm::vec3 camRight = glm::vec3(oocRasterCst.view[0][0], oocRasterCst.view[1][0], oocRasterCst.view[2][0]);
    glm::vec3 camUp    = glm::vec3(oocRasterCst.view[0][1], oocRasterCst.view[1][1], oocRasterCst.view[2][1]);
    glm::vec3 camFront = -glm::vec3(oocRasterCst.view[0][2], oocRasterCst.view[1][2], oocRasterCst.view[2][2]);

    return glm::normalize(p.x * camRight * halfFovTan +
                          p.y * camUp    * fovTan     +
                          camFront);
}

__global__ void sphereRasterOocKernel(
    const glm::vec4*    __restrict__ activeAtoms,
    unsigned int                     activeCount,
    unsigned int*       __restrict__ depthBuffer,
    cudaSurfaceObject_t              outputImage)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= activeCount) return;

    glm::vec4 atom = activeAtoms[idx];
    glm::vec3 pos  = glm::vec3(atom);
    float     rad  = atom.w;

    // Project to screen-space bounding box
    glm::vec4 camSpace4 = oocRasterCst.view * glm::vec4(pos, 1.0f);
    glm::vec3 cs = glm::vec3(camSpace4);
    glm::vec3 ncs = glm::normalize(cs);
    glm::vec3 imp = cs - ncs * rad;

    float dist = glm::length(cs) + 1e-6f;
    float sinA = rad / dist;
    float tanA = tanf(asinf(fminf(sinA, 0.999f)));
    float qs   = tanA * glm::length(imp);

    glm::vec3 up(0.0f, 1.0f, 0.0f);
    glm::vec3 u = glm::normalize(glm::cross(ncs, up));
    if (glm::length(u) < 0.001f) u = glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 v = glm::cross(u, ncs) * qs;
    u *= qs;

    glm::vec3 corners[4] = {
        imp + u + v, imp - u + v, imp + u - v, imp - u - v
    };

    glm::vec2 minC(1e6f), maxC(-1e6f);
    for (int i = 0; i < 4; i++) {
        glm::vec4 clip = oocRasterCst.proj * glm::vec4(corners[i], 1.0f);
        float iw = fdividef(1.0f, clip.w);
        float cx = clip.x * iw;
        float cy = clip.y * iw;
        minC.x = fminf(minC.x, cx); minC.y = fminf(minC.y, cy);
        maxC.x = fmaxf(maxC.x, cx); maxC.y = fmaxf(maxC.y, cy);
    }

    int sMinX = max(0, (int)floorf((minC.x * 0.5f + 0.5f) * oocRasterCst.screenWidth));
    int sMinY = max(0, (int)floorf((minC.y * 0.5f + 0.5f) * oocRasterCst.screenHeight));
    int sMaxX = min(oocRasterCst.screenWidth,  (int)ceilf((maxC.x * 0.5f + 0.5f) * oocRasterCst.screenWidth));
    int sMaxY = min(oocRasterCst.screenHeight, (int)ceilf((maxC.y * 0.5f + 0.5f) * oocRasterCst.screenHeight));

    for (int py = sMinY; py < sMaxY; py++) {
        for (int px = sMinX; px < sMaxX; px++) {
            glm::vec3 rd = computeRayOoc(px, py);
            float t = iSphereOoc(oocRasterCst.cameraPos, rd, pos, rad);
            if (t > 0.0f) {
                glm::vec3 hit = oocRasterCst.cameraPos + rd * t;
                glm::vec4 hitClip = oocRasterCst.proj * (oocRasterCst.view * glm::vec4(hit, 1.0f));
                float depth = hitClip.z / hitClip.w;
                unsigned int depthU = __float_as_uint(depth);
                int pixelIdx = py * oocRasterCst.screenWidth + px;
                if (atomicMin(&depthBuffer[pixelIdx], depthU) != depthU) {
                    glm::vec3 normal = glm::normalize(hit - pos);
                    float lambert = fmaxf(0.0f, glm::dot(normal, -glm::normalize(rd)));
                    glm::vec3 color = glm::vec3(0.01f, 1.0f, 0.05f) * lambert * 0.9f;
                    uchar4 pixel = make_uchar4(
                        static_cast<unsigned char>(color.x * 255.0f),
                        static_cast<unsigned char>(color.y * 255.0f),
                        static_cast<unsigned char>(color.z * 255.0f), 255);
                    surf2Dwrite(pixel, outputImage, px * sizeof(uchar4), py);
                }
            }
        }
    }
}

extern "C" void launchSphereRasterOoc(
    const glm::vec4*    d_activeAtoms,
    unsigned int        activeCount,
    unsigned int*       d_depthBuffer,
    cudaSurfaceObject_t outputImage,
    cudaStream_t        stream)
{
    if (activeCount == 0) return;
    dim3 block(256);
    dim3 grid((activeCount + block.x - 1) / block.x);
    sphereRasterOocKernel<<<grid, block, 0, stream>>>(
        d_activeAtoms, activeCount, d_depthBuffer, outputImage);
}
