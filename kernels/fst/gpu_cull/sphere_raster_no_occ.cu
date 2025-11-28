// depth_occlusion_kernel.cu
#define GLM_FORCE_CUDA
#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_INLINE
#define CUDA_VERSION 13000

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <nvtx3/nvToolsExt.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>
#include <cstdint>
#include <stdio.h>
#include "utils/scene_descriptor.h"

#define pixelsToCheck 5

// ========================================================
// -----------------------------
// CONSTANT MEMORY (per-frame)
// -----------------------------

struct Constants {
    glm::mat4 view;
    glm::mat4 proj;
    glm::vec4 frustumPlanes[6];
    glm::vec3 front, up, right, cameraPos;
    int screenW, screenH;
    float fov;
    int sphereCount;
    int benchmark;
};

static __constant__ Constants cst;

struct FramePixelsCount {
    unsigned int frames;
    unsigned int pixels;
};

// ========================================================
// Device helpers
__device__ inline unsigned int floatToUintBits(float f) { return __float_as_uint(f); }
__device__ inline float uintToFloatBits(unsigned int u) { return __uint_as_float(u); }

__device__ inline int clampi(int v, int lo, int hi) {
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

// iSphere (returns t or -1)
static __device__ float iSphere(const glm::vec3& ro, const glm::vec3& rd, const glm::vec3& sph, float radius) {
    glm::vec3 oc = ro - sph;
    float b = glm::dot(oc, rd);
    float c = glm::dot(oc, oc) - radius * radius;
    float h = fmaf(b, b, -c);
    if (h < 0.0f) return -1.0f;
    return -b - sqrtf(h);
}

// makeScreenRayCasting (returns rayStart, dx, dy)
struct ScreenRayCasting {
    glm::vec3 rayStart;
    glm::vec3 dx;
    glm::vec3 dy;
};

__device__ inline ScreenRayCasting makeScreenRayCasting(
    const float& fovRad,
    const float& fovTan,
    const float& halfFovTan)
{
    glm::vec3 halfFovTan_right = halfFovTan * cst.right;
    glm::vec3 fovTan_up = fovTan * cst.up;
    
    glm::vec3 corner00 = glm::normalize(-halfFovTan_right + (-fovTan_up) + cst.front);
    glm::vec3 corner10 = glm::normalize(( halfFovTan_right) + (-fovTan_up) + cst.front);
    glm::vec3 corner01 = glm::normalize((-halfFovTan_right) + ( fovTan_up) + cst.front);

    glm::vec3 wCorner00 = glm::mat3(cst.view) * corner00;
    glm::vec3 wCorner10 = glm::mat3(cst.view) * corner10;
    glm::vec3 wCorner01 = glm::mat3(cst.view) * corner01;

    glm::vec3 dx = (wCorner10 - wCorner00) / float(cst.screenW);
    glm::vec3 dy = (wCorner01 - wCorner00) / float(cst.screenH);
    glm::vec3 rayStart = wCorner00;

    return { rayStart, dx, dy };
}

// texelFetch-like read from cudaTextureObject_t (point sampling)
__device__ inline float texelFetchDepth(cudaTextureObject_t tex, int x, int y) 
{
    // use integer coords + 0.5f to sample exact texel (readMode = cudaReadModeElementType)
    return tex2D<float>(tex, float(x) + 0.5f, float(y) + 0.5f);
}

// frustum plane test (plane: vec4(nx,ny,nz,w)), sphere: vec4(x,y,z,r)
__device__ inline bool isOnOrForwardOfPlane(const glm::vec4& plane, const glm::vec4& sphPosR) {
    float d = glm::dot(glm::vec3(plane), glm::vec3(sphPosR)) - plane.w;
    return d >= -sphPosR.w;
}

__device__ inline bool isSphereInside(const glm::vec4& sphPosR)
{
    return isOnOrForwardOfPlane(cst.frustumPlanes[0], sphPosR) &&
           isOnOrForwardOfPlane(cst.frustumPlanes[1], sphPosR) &&
           isOnOrForwardOfPlane(cst.frustumPlanes[2], sphPosR) &&
           isOnOrForwardOfPlane(cst.frustumPlanes[3], sphPosR) &&
           isOnOrForwardOfPlane(cst.frustumPlanes[4], sphPosR) &&
           isOnOrForwardOfPlane(cst.frustumPlanes[5], sphPosR);
}

// compute eye ray direction (equivalent to computeRd in GLSL)
__device__ inline glm::vec3 computeRd(int px, int py, const float& fovTan, const float& halfFovTan)
{
    glm::vec2 p = ( -glm::vec2(cst.screenW, cst.screenH) + 2.0f * glm::vec2(px, py) ) / glm::vec2(cst.screenW, cst.screenH);
    return glm::normalize(p.x * cst.right * halfFovTan + p.y * cst.up * fovTan + cst.front);
}

// onSphDepth same math as GLSL
__device__ inline float onSphDepth(const glm::vec3& rd, int px, int py, const glm::vec3& spherePos, float r,
                            const glm::vec3& rayStart, const glm::vec3& dx, const glm::vec3& dy)
{
    float h = iSphere(cst.cameraPos, rd, spherePos, r);
    glm::vec3 hit = (rayStart + float(px) * dx + float(py) * dy) * h;
    // project depth like GLSL: (hit.z * proj[2].z + proj[3].z) / -hit.z
    float depth = (fmaf(hit.z, cst.proj[2][2], cst.proj[3][2])) / -hit.z;
    return depth;
}

__device__ inline glm::vec2 safeMin(const glm::vec2& a, const glm::vec2& b) {
    return glm::vec2(fminf(a.x, b.x), fminf(a.y, b.y));
}

__device__ inline glm::vec2 safeMax(const glm::vec2& a, const glm::vec2& b) {
    return glm::vec2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
}

// ========================================================
// Kernel: one thread per sphere
// ========================================================
__global__ void sphereRasterKernelNoOcc(
    // storage buffers (lineal)
    glm::vec4* __restrict__ spheres,
    unsigned int* __restrict__ depthBuffer,            // in uint bits of float depth, length = screenW*screenH
    // counters
    unsigned int* frustCullcounter,
    // outputs (linear) -- optional, can pass nullptr if unused
    cudaSurfaceObject_t outputImage
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= cst.sphereCount) return;

    
    glm::vec4 posr = spheres[idx];
    // glm::vec4 posr = s.positionr;

    // Frustum test
    if (!isSphereInside(posr)) return;

    if (cst.benchmark == 1) atomicAdd(frustCullcounter, 1u);

    // compute bbox etc.
    // camera-space sphere
    glm::vec4 worldPos = glm::vec4(posr.x, posr.y, posr.z, 1.0f);
    glm::vec4 camSpace4 = cst.view * worldPos;
    glm::vec3 cameraSpaceSphere = glm::vec3(camSpace4.x, camSpace4.y, camSpace4.z);
    glm::vec3 normCamSpaceSphere = glm::normalize(cameraSpaceSphere);
    glm::vec3 camImposPos = cameraSpaceSphere - normCamSpaceSphere * posr.w;

    // bbox
    const float sinAngle = posr.w / (glm::length(cameraSpaceSphere) + 1e-6f);
    const float tanAngle = tanf(asinf(sinAngle));
    const float quadScale = tanAngle * glm::length(camImposPos);

    glm::vec3 impU = glm::normalize(glm::cross(normCamSpaceSphere, cst.up));
    glm::vec3 impV = glm::cross(impU, normCamSpaceSphere) * quadScale;
    impU *= quadScale;

    // Corners
    glm::vec3 corners[4];
    corners[0] = camImposPos + impU + impV;
    corners[1] = camImposPos - impU + impV;
    corners[2] = camImposPos + impU - impV;
    corners[3] = camImposPos - impU - impV;

    // Project + normalize
    glm::vec2 minC(1e6f), maxC(-1e6f);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        glm::vec4 clip = cst.proj * glm::vec4(corners[i], 1.0f);
        float iw = 1.0f / clip.w;
        float x = clip.x * iw;
        float y = clip.y * iw;
        minC.x = fminf(minC.x, x);
        minC.y = fminf(minC.y, y);
        maxC.x = fmaxf(maxC.x, x);
        maxC.y = fmaxf(maxC.y, y);
    }

    // Convert to screen coordinates (fused ops)
    int screenMinX = __float2int_rd(fmaf(minC.x, 0.5f, 0.5f) * cst.screenW);
    int screenMaxX = __float2int_ru(fmaf(maxC.x, 0.5f, 0.5f) * cst.screenW);
    int screenMinY = __float2int_rd(fmaf(minC.y, 0.5f, 0.5f) * cst.screenH);
    int screenMaxY = __float2int_ru(fmaf(maxC.y, 0.5f, 0.5f) * cst.screenH);

    float difx = float(screenMaxX - screenMinX);
    float dify = float(screenMaxY - screenMinY);
    if (difx * dify <= 2.0f) return;

    float aspectRatio = float(cst.screenW) / float(cst.screenH);
    float fovRad = glm::radians(cst.fov);
    float fovTan = tanf(fovRad * 0.5f);
    float halfFovTan = fovTan * aspectRatio;

    // ray casting helpers
    ScreenRayCasting scrc = makeScreenRayCasting(fovRad, fovTan, halfFovTan);
    glm::vec3 dx = scrc.dx;
    glm::vec3 dy = scrc.dy;
    glm::vec3 rayStart = scrc.rayStart;

    // rasterize inside bbox
    int minx = max(0, screenMinX);
    int maxx = min(screenMaxX, cst.screenW);
    int miny = max(0, screenMinY);
    int maxy = min(screenMaxY, cst.screenH);

    float proj22 = cst.proj[2][2];
    float proj32 = cst.proj[3][2];

    for (int px = minx; px < maxx; ++px) {
        glm::vec3 rayColStart = rayStart + float(px) * dx;
        bool finishedLine = false;
        for (int py = miny; py < maxy; ++py) {
            // surf2Dwrite(make_uchar4(255, 255, 255, 255), outputImage,
            // static_cast<unsigned int>(px) * sizeof(uchar4),
            // static_cast<unsigned int>(py));
            glm::vec3 rd = computeRd(px, py, fovTan, halfFovTan);
            float t = iSphere(glm::vec3(cst.cameraPos), rd, glm::vec3(posr), posr.w);
            if (t > 0.0f) {
                finishedLine = true;
                glm::vec3 hit = (rayColStart + float(py) * dy) * t;
                float depth = fmaf(hit.z, proj22, proj32) / -hit.z;

                unsigned int depthU = floatToUintBits(depth);
                int index = px + cst.screenW * py;

                // atomicMin over uint bits (we store depth as float bits in uint)
                unsigned int old = atomicMin(&depthBuffer[index], depthU);

                // if we won (old > depth)
                if (uintToFloatBits(old) > depth) {
                    // own the pixel
                    glm::vec3 normal = glm::normalize(glm::vec3(cst.cameraPos) + rd * t - glm::vec3(posr));
                    float lambert = glm::max(0.0f, glm::dot(normal, -glm::normalize(rd * t)));
                    glm::vec3 color = glm::vec3(atomsColor.x, atomsColor.y, atomsColor.z) * lambert * diffuse;
                    uchar4 ucharColor = make_uchar4(color.x*255, color.y*255, color.z*255, 255); 
                    surf2Dwrite(ucharColor, outputImage, px * sizeof(uchar4), py); 
                }
            } else if (finishedLine) {
                break;
            }
        }
    }
}


extern "C" void sphereRasterNoOcc(
    glm::vec4* d_spheres,
    int c_sphereCount,
    glm::mat4 c_view,
    glm::mat4 c_proj,
    glm::vec4 c_frustumTopFace,
    glm::vec4 c_frustumBottomFace,
    glm::vec4 c_frustumRightFace,
    glm::vec4 c_frustumLeftFace,
    glm::vec4 c_frustumFarFace,
    glm::vec4 c_frustumNearFace,
    glm::vec3 c_front,
    glm::vec3 c_up,
    glm::vec3 c_right,
    glm::vec3 c_cameraPos,
    int c_screenW,
    int c_screenH,
    float c_fov,
    unsigned int* d_depthBuffer,
    unsigned int* d_frustCullcounter,
    int c_benchmark,
    cudaSurfaceObject_t outputImage,
    cudaStream_t& stream,
    int workGroupSizeX
)
{

    Constants c_constants = {
    c_view,
    c_proj,
    {c_frustumTopFace,
    c_frustumBottomFace,
    c_frustumRightFace,
    c_frustumLeftFace,
    c_frustumFarFace,
    c_frustumNearFace},
    c_front,
    c_up,
    c_right,
    c_cameraPos,
    c_screenW,
    c_screenH,
    c_fov,
    c_sphereCount,
    c_benchmark
    };
    // copy constants to symbol memory (per-frame)
    cudaMemcpyToSymbolAsync(cst, &c_constants, sizeof(Constants), 0, cudaMemcpyHostToDevice, stream);

    nvtxRangePushA("Sphere Raster No Occ Kernel");
    dim3 block(workGroupSizeX);
    dim3 grid((c_sphereCount + block.x - 1) / block.x);
    sphereRasterKernelNoOcc<<<grid, block, 0, stream>>>(
        d_spheres,
        d_depthBuffer,
        d_frustCullcounter,
        outputImage
    );
    nvtxRangePop();
}