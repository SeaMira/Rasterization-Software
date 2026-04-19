/**
 * @file sphere_raster_ooc.cu
 * @brief Direct sphere rasterisation for the out-of-core pipeline.
 *
 * Drawn-atom stats (d_rasterAtomCount): only atoms that perform at least one
 * surf2Dwrite after winning the depth test (not merely entering the screen bbox).
 *
 * Screen-space bounds: AABB in NDC of the silhouette circle (24 rim samples +
 * projected center + small NDC pad), tighter than the old impostor quad.
 * Then per-pixel ray-sphere like smallSphereRasterKernel; uses rayStart/dx/dy.
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
    cudaSurfaceObject_t              outputImage,
    unsigned int*       __restrict__ d_rasterAtomCount)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= activeCount) return;

    glm::vec4 atom = activeAtoms[idx];
    glm::vec3 pos  = glm::vec3(atom);
    float     rad  = atom.w;

    glm::vec4 camSpace4 = oocRasterCst.view * glm::vec4(pos, 1.0f);
    glm::vec3 c = glm::vec3(camSpace4);

    // Silhouette circle: plane through sphere center, perpendicular to view ray from
    // eye (origin). Bounding its perspective projection with an axis-aligned NDC box
    // is much tighter than the old impostor quad (camImposPos + scaled offsets),
    // which inflated the pixel loop for nearby spheres.
    float lenC2 = glm::dot(c, c);
    if (lenC2 < 1e-12f)
        return;
    float invLenC = rsqrtf(lenC2);
    glm::vec3 viewDir = c * invLenC;

    glm::vec3 upRef(0.0f, 1.0f, 0.0f);
    glm::vec3 axisU = glm::cross(upRef, viewDir);
    float uLen2 = glm::dot(axisU, axisU);
    if (uLen2 < 1e-10f)
        axisU = glm::vec3(1.0f, 0.0f, 0.0f);
    else
        axisU = axisU * rsqrtf(uLen2);
    glm::vec3 axisV = glm::cross(viewDir, axisU);

    // View space from glm::lookAt: camera faces -Z. Visible half-space beyond the
    // near clip plane: z < -nearPlane. Skip if any silhouette sample is behind the
    // eye, between eye and near, or has non-positive homogeneous w after proj.
    const float nearClipZ = -oocRasterCst.nearPlane;

    glm::vec2 minC(1e6f), maxC(-1e6f);

    // 24 samples on the silhouette circle (every 15°). Finer than 8 avoids
    // under-shooting the true NDC AABB of the projected ellipse; small NDC pad below
    // keeps the bbox conservative vs cracks.
    const int numSamples = 12;
    float numSamplesFloat = static_cast<float>(numSamples);
    for (int k = 0; k < numSamples; k++) {
        float ang = (2.0f * 3.14159265f / numSamplesFloat) * static_cast<float>(k);
        float ca = cosf(ang);
        float sa = sinf(ang);
        glm::vec3 p = c + rad * (ca * axisU + sa * axisV);
        if (p.z >= nearClipZ)
            return;
        glm::vec4 clip = oocRasterCst.proj * glm::vec4(p, 1.0f);
        if (clip.w <= 1e-6f)
            return;
        float iw = __fdividef(1.0f, clip.w);
        float x = clip.x * iw;
        float y = clip.y * iw;
        minC.x = fminf(minC.x, x);
        minC.y = fminf(minC.y, y);
        maxC.x = fmaxf(maxC.x, x);
        maxC.y = fmaxf(maxC.y, y);
    }

    if (c.z >= nearClipZ)
        return;
    {
        glm::vec4 clipC = oocRasterCst.proj * glm::vec4(c, 1.0f);
        if (clipC.w <= 1e-6f)
            return;
        float iw = __fdividef(1.0f, clipC.w);
        float x = clipC.x * iw;
        float y = clipC.y * iw;
        minC.x = fminf(minC.x, x);
        minC.y = fminf(minC.y, y);
        maxC.x = fmaxf(maxC.x, x);
        maxC.y = fmaxf(maxC.y, y);
    }

    {
        float padX = 2.0f / fmaxf(1.0f, float(oocRasterCst.screenWidth));
        float padY = 2.0f / fmaxf(1.0f, float(oocRasterCst.screenHeight));
        minC.x -= padX;
        minC.y -= padY;
        maxC.x += padX;
        maxC.y += padY;
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

    // Count only atoms that actually shade at least one pixel (won depth vs buffer).
    unsigned int wroteVisiblePixel = 0u;

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
                    wroteVisiblePixel = 1u;
                }
            }
        }
    }

    if (wroteVisiblePixel != 0u && d_rasterAtomCount != nullptr)
        atomicAdd(d_rasterAtomCount, 1u);
}

extern "C" void launchSphereRasterOoc(
    const glm::vec4*    d_activeAtoms,
    unsigned int       activeCount,
    unsigned int*      d_depthBuffer,
    cudaSurfaceObject_t outputImage,
    unsigned int*      d_rasterAtomCount,
    cudaStream_t       stream)
{
    if (activeCount == 0) return;
    dim3 block(256);
    dim3 grid((activeCount + block.x - 1) / block.x);
    sphereRasterOocKernel<<<grid, block, 0, stream>>>(
        d_activeAtoms, activeCount, d_depthBuffer, outputImage, d_rasterAtomCount);
}
