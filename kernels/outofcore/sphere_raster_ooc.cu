/**
 * @file sphere_raster_ooc.cu
 * @brief Direct sphere rasterisation for the out-of-core pipeline.
 *
 * Same algorithm as smallSphereRasterKernel in small_entity_raster.cu:
 * one thread per atom, projects screen-space bounding box, ray-tests each
 * pixel for exact sphere intersection. Uses pre-computed rayStart/dx/dy.
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
                                   const glm::vec3& center, float radius) 
                                   {
    glm::vec3 oc = ro - center;
    float b = glm::dot(oc, rd);
    float c = glm::dot(oc, oc) - radius * radius;
    float h = b * b - c;
    if (h < 0.0f) return -1.0f;
    return -b - sqrtf(h);
}

__device__ inline glm::vec3 fastNormalize(const glm::vec3& v) {
    return v * rsqrtf(glm::dot(v, v));
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

    glm::vec4 camSpace4 = oocRasterCst.view * glm::vec4(pos, 1.0f);
    glm::vec3 cameraSpaceSphere = glm::vec3(camSpace4);
    glm::vec3 normCamSpaceSphere = glm::normalize(cameraSpaceSphere);
    glm::vec3 camImposPos = cameraSpaceSphere - normCamSpaceSphere * rad;

    float dist = glm::length(cameraSpaceSphere) + 1e-6f;
    float sinAngle = __fdividef(rad, dist);
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
        glm::vec4 clip = oocRasterCst.proj * glm::vec4(corners[i], 1.0f);
        float iw = __fdividef(1.0f, clip.w);
        float x = clip.x * iw;
        float y = clip.y * iw;
        minC.x = fminf(minC.x, x); minC.y = fminf(minC.y, y);
        maxC.x = fmaxf(maxC.x, x); maxC.y = fmaxf(maxC.y, y);
    }

    int screenMinX = max(0, __float2int_rd(fmaf(minC.x, 0.5f, 0.5f) * oocRasterCst.screenWidth));
    int screenMinY = max(0, __float2int_rd(fmaf(minC.y, 0.5f, 0.5f) * oocRasterCst.screenHeight));
    int screenMaxX = min(oocRasterCst.screenWidth,  __float2int_ru(fmaf(maxC.x, 0.5f, 0.5f) * oocRasterCst.screenWidth));
    int screenMaxY = min(oocRasterCst.screenHeight, __float2int_ru(fmaf(maxC.y, 0.5f, 0.5f) * oocRasterCst.screenHeight));

    float difx = float(screenMaxX - screenMinX);
    float dify = float(screenMaxY - screenMinY);
    if (difx * dify <= 2.0f) return;

    float proj22 = oocRasterCst.proj[2][2];
    float proj32 = oocRasterCst.proj[3][2];

    for (int py = screenMinY; py < screenMaxY; py++) {
        glm::vec3 rayColStart = oocRasterCst.rayStart + (float)py * oocRasterCst.dy;
        for (int px = screenMinX; px < screenMaxX; px++) {
            glm::vec3 ray = rayColStart + (float)px * oocRasterCst.dx;
            glm::vec3 rd = fastNormalize(ray.x * oocRasterCst.right + ray.y * oocRasterCst.up - ray.z * oocRasterCst.front);
            float t = iSphereOoc(oocRasterCst.cameraPos, rd, pos, rad);

            if (t > 0.0f) {
                glm::vec3 hit = ray * t;
                float depth = __fdividef(fmaf(hit.z, proj22, proj32), -hit.z);
                unsigned int depthU = __float_as_uint(depth);

                int pixelIdx = py * oocRasterCst.screenWidth + px;
                unsigned int old = atomicMin(&depthBuffer[pixelIdx], depthU);
                if (__uint_as_float(old) > depth) {
                    glm::vec3 normal = fastNormalize(oocRasterCst.cameraPos + rd * t - pos);
                    float lambert = fmaxf(0.0f, glm::dot(normal, -rd));
                    glm::vec3 color = glm::vec3(oocRasterCst.atomsColor.x, oocRasterCst.atomsColor.y, oocRasterCst.atomsColor.z) * lambert * oocRasterCst.diffuse;
                    uchar4 ucharColor = make_uchar4(
                        (unsigned char)(color.x * 255.0f),
                        (unsigned char)(color.y * 255.0f),
                        (unsigned char)(color.z * 255.0f), 255);
                    surf2Dwrite(ucharColor, outputImage, px * sizeof(uchar4), py);
                }
            }
        }
    }
}

extern "C" void launchSphereRasterOoc(
    const glm::vec4*    d_activeAtoms,
    unsigned int       activeCount,
    unsigned int*      d_depthBuffer,
    cudaSurfaceObject_t outputImage,
    cudaStream_t       stream)
{
    if (activeCount == 0) return;
    dim3 block(256);
    dim3 grid((activeCount + block.x - 1) / block.x);
    sphereRasterOocKernel<<<grid, block, 0, stream>>>(
        d_activeAtoms, activeCount, d_depthBuffer, outputImage);
}
