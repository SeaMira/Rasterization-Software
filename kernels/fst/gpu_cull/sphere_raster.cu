// depth_occlusion_kernel.cu
#define GLM_FORCE_CUDA
#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_INLINE
#define CUDA_VERSION 13000

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>
#include <cstdint>

// ========================================================
// Types (mirror GLSL)
// struct Sphere {
//     glm::vec4 positionr; // (x,y,z,r)
// };

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
__device__ float iSphere(const glm::vec3& ro, const glm::vec3& rd, const glm::vec3& sph, float radius) {
    glm::vec3 oc = ro - sph;
    float b = glm::dot(oc, rd);
    float c = glm::dot(oc, oc) - radius * radius;
    float h = b*b - c;
    if (h < 0.0f) return -1.0f;
    return -b - sqrtf(h);
}

// makeScreenRayCasting (returns rayStart, dx, dy)
struct ScreenRayCasting {
    glm::vec3 rayStart;
    glm::vec3 dx;
    glm::vec3 dy;
};

__device__ ScreenRayCasting makeScreenRayCasting(
    const glm::mat4& view,
    const glm::vec3& front,
    const glm::vec3& up,
    const glm::vec3& right,
    int screenW,
    int screenH,
    float fov)
{
    float aspectRatio = float(screenW) / float(screenH);
    float fovRad = glm::radians(fov);
    float fovTan = tanf(fovRad * 0.5f);
    float halfFovTan = fovTan * aspectRatio;

    glm::vec3 corner00 = glm::normalize((-halfFovTan) * right + (-fovTan) * up + front);
    glm::vec3 corner10 = glm::normalize(( halfFovTan) * right + (-fovTan) * up + front);
    glm::vec3 corner01 = glm::normalize((-halfFovTan) * right + ( fovTan) * up + front);

    glm::vec3 wCorner00 = glm::mat3(view) * corner00;
    glm::vec3 wCorner10 = glm::mat3(view) * corner10;
    glm::vec3 wCorner01 = glm::mat3(view) * corner01;

    glm::vec3 dx = (wCorner10 - wCorner00) / float(screenW);
    glm::vec3 dy = (wCorner01 - wCorner00) / float(screenH);
    glm::vec3 rayStart = wCorner00;

    return { rayStart, dx, dy };
}

// texelFetch-like read from cudaTextureObject_t (point sampling)
__device__ inline float texelFetchDepth(cudaTextureObject_t tex, int x, int y) {
    // use integer coords + 0.5f to sample exact texel (readMode = cudaReadModeElementType)
    return tex2D<float>(tex, float(x) + 0.5f, float(y) + 0.5f);
}

// billboard pre-test: samples 5 texels from downsampled texture
__device__ bool isSphereBillboardVisible(
    cudaTextureObject_t downsampleTex,
    int screenW, int screenH,
    int xcoords[], int ycoords[], float pixelDepths[], int pixelsToCheck = 5)
{
    int dsW = max(1, screenW / 16);
    int dsH = max(1, screenH / 16);

    // sample mapped coords (like original)
    float downvals[5];
    for (int i = 0; i < pixelsToCheck; ++i) {
        int tx = int(float(xcoords[i]) * float(dsW) / float(screenW));
        int ty = int(float(ycoords[i]) * float(dsH) / float(screenH));
        tx = clampi(tx, 0, dsW - 1);
        ty = clampi(ty, 0, dsH - 1);
        downvals[i] = texelFetchDepth(downsampleTex, tx, ty);
    }

    for (int i = 0; i < pixelsToCheck; ++i) {
        float pd = pixelDepths[i];
        if (pd <= downvals[0] || pd <= downvals[1] || pd <= downvals[2] || pd <= downvals[3] || pd <= downvals[4])
            return true;
    }
    return false;
}

// frustum plane test (plane: vec4(nx,ny,nz,w)), sphere: vec4(x,y,z,r)
__device__ inline bool isOnOrForwardOfPlane(const glm::vec4& plane, const glm::vec4& sphPosR) {
    float d = glm::dot(glm::vec3(plane), glm::vec3(sphPosR)) - plane.w;
    return d >= -sphPosR.w;
}

__device__ inline bool isSphereInside(const glm::vec4& sphPosR,
                                      const glm::vec4& frustumLeft,
                                      const glm::vec4& frustumRight,
                                      const glm::vec4& frustumFar,
                                      const glm::vec4& frustumNear,
                                      const glm::vec4& frustumTop,
                                      const glm::vec4& frustumBottom)
{
    return isOnOrForwardOfPlane(frustumLeft, sphPosR) &&
           isOnOrForwardOfPlane(frustumRight, sphPosR) &&
           isOnOrForwardOfPlane(frustumFar, sphPosR) &&
           isOnOrForwardOfPlane(frustumNear, sphPosR) &&
           isOnOrForwardOfPlane(frustumTop, sphPosR) &&
           isOnOrForwardOfPlane(frustumBottom, sphPosR);
}

// compute eye ray direction (equivalent to computeRd in GLSL)
__device__ glm::vec3 computeRd(int px, int py, int screenW, int screenH, const glm::vec3& right, const glm::vec3& up, const glm::vec3& front, float fov)
{
    glm::vec2 p = ( -glm::vec2(screenW, screenH) + 2.0f * glm::vec2(px, py) ) / glm::vec2(screenW, screenH);
    float fovRad = glm::radians(fov);
    float fovTan = tanf(fovRad * 0.5f);
    float halfFovTan = fovTan * (float(screenW) / float(screenH));
    return glm::normalize(p.x * right * halfFovTan + p.y * up * fovTan + front);
}

// onSphDepth same math as GLSL
__device__ float onSphDepth(const glm::vec3& rd, int px, int py, const glm::vec3& spherePos, float r,
                            const glm::vec3& rayStart, const glm::vec3& dx, const glm::vec3& dy,
                            const glm::mat4& proj, const glm::vec3& cameraPos)
{
    float h = iSphere(cameraPos, rd, spherePos, r);
    glm::vec3 hit = (rayStart + float(px) * dx + float(py) * dy) * h;
    // project depth like GLSL: (hit.z * proj[2].z + proj[3].z) / -hit.z
    float proj2z = proj[2][2]; // assumes column-major glm (access as proj[col][row])
    float proj3z = proj[3][2];
    float depth = (hit.z * proj2z + proj3z) / -hit.z;
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
__global__ void sphereRasterKernel(
    // buffers (device pointers)
    glm::vec4* __restrict__ spheres,
    int sphereCount,
    // frame / view data
    glm::mat4 view,
    glm::mat4 proj,
    glm::vec4 frustumTopFace,
    glm::vec4 frustumBottomFace,
    glm::vec4 frustumRightFace,
    glm::vec4 frustumLeftFace,
    glm::vec4 frustumFarFace,
    glm::vec4 frustumNearFace,
    glm::vec3 front,
    glm::vec3 up,
    glm::vec3 rightVec,
    glm::vec3 cameraPos,
    int screenW,
    int screenH,
    float fov,
    // storage buffers (lineal)
    unsigned int* depthBuffer,            // in uint bits of float depth, length = screenW*screenH
    unsigned int* pixelOwnershipBuffer,   // length = screenW*screenH
    FramePixelsCount* visibilityFrameBuffer,
    unsigned int visibilityFrameBufferIndexOffset,
    // counters
    unsigned int* frustCullcounter,
    unsigned int* occCullcounter,
    int benchmark,
    // outputs (linear) -- optional, can pass nullptr if unused
    cudaSurfaceObject_t outputImage,         // RGBA8 linear buffer [screenW*screenH*4]
    // downsample texture object for billboard test
    cudaTextureObject_t downsampleTex
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= sphereCount) return;

    glm::vec4 posr = spheres[idx];
    // glm::vec4 posr = s.positionr;

    // Frustum test
    if (!isSphereInside(posr, frustumLeftFace, frustumRightFace, frustumFarFace,
                        frustumNearFace, frustumTopFace, frustumBottomFace))
    {
        // mark zero pixels like shader did
        unsigned int visId = visibilityFrameBufferIndexOffset + idx;
        visibilityFrameBuffer[visId].pixels = 0;

        return;
    }

    if (benchmark == 1) atomicAdd(frustCullcounter, 1u);

    // compute bbox etc.
    // camera-space sphere
    glm::vec4 worldPos = glm::vec4(posr.x, posr.y, posr.z, 1.0f);
    glm::vec4 camSpace4 = view * worldPos;
    glm::vec3 cameraSpaceSphere = glm::vec3(camSpace4.x, camSpace4.y, camSpace4.z);
    glm::vec3 normCamSpaceSphere = glm::normalize(cameraSpaceSphere);
    glm::vec3 camImposPos = cameraSpaceSphere - normCamSpaceSphere * posr.w;

    // bbox
    const float sinAngle = posr.w / (glm::length(cameraSpaceSphere) + 1e-6f);
    const float tanAngle = tanf(asinf(sinAngle));
    const float quadScale = tanAngle * glm::length(camImposPos);

    glm::vec3 impU = glm::normalize(glm::cross(normCamSpaceSphere, up));
    glm::vec3 impV = glm::cross(impU, normCamSpaceSphere) * quadScale;
    impU *= quadScale;

    glm::vec3 upRightCorner = camImposPos + impU + impV;
    glm::vec3 upLeftCorner  = camImposPos - impU + impV;
    glm::vec3 downRightCorner = camImposPos + impU - impV;
    glm::vec3 downLeftCorner  = camImposPos - impU - impV;

    glm::vec4 upRight = proj * glm::vec4(upRightCorner, 1.0f);
    glm::vec4 upLeft  = proj * glm::vec4(upLeftCorner, 1.0f);
    glm::vec4 downRight = proj * glm::vec4(downRightCorner, 1.0f);
    glm::vec4 downLeft  = proj * glm::vec4(downLeftCorner, 1.0f);

    auto ndc = [](const glm::vec4& p)->glm::vec3 {
        return glm::vec3(p.x / p.w, p.y / p.w, p.z / p.w);
    };

    glm::vec3 ndcUR = ndc(upRight);
    glm::vec3 ndcUL = ndc(upLeft);
    glm::vec3 ndcDR = ndc(downRight);
    glm::vec3 ndcDL = ndc(downLeft);

    glm::vec2 minCorner = safeMin(
        safeMin(glm::vec2(ndcUR.x, ndcUR.y), glm::vec2(ndcUL.x, ndcUL.y)),
        safeMin(glm::vec2(ndcDR.x, ndcDR.y), glm::vec2(ndcDL.x, ndcDL.y))
    );
    glm::vec2 maxCorner = safeMax(
        safeMax(glm::vec2(ndcUR.x, ndcUR.y), glm::vec2(ndcUL.x, ndcUL.y)),
        safeMax(glm::vec2(ndcDR.x, ndcDR.y), glm::vec2(ndcDL.x, ndcDL.y))
    );

    int screenMinX = int((minCorner.x * 0.5f + 0.5f) * float(screenW));
    int screenMinY = int((minCorner.y * 0.5f + 0.5f) * float(screenH));
    int screenMaxX = int((maxCorner.x * 0.5f + 0.5f) * float(screenW));
    int screenMaxY = int((maxCorner.y * 0.5f + 0.5f) * float(screenH));

    float difx = float(screenMaxX - screenMinX);
    float dify = float(screenMaxY - screenMinY);
    if (difx * dify <= 2.0f) {
        visibilityFrameBuffer[visibilityFrameBufferIndexOffset + idx].pixels = 0;
        // atomicAdd(frustCullcounter, 1u);
        return;
    }

    // Prepare billboard samples (five)
    int midx = clampi((screenMinX + screenMaxX) / 2, 0, screenW);
    int midy = clampi((screenMinY + screenMaxY) / 2, 0, screenH);

    int xcoords[5] = { int(midx - difx * 0.49f), int(midx + difx * 0.49f), midx, midx, (screenMinX + screenMaxX) / 2 };
    int ycoords[5] = { midy, midy, int(midy - dify * 0.49f), int(midy + dify * 0.49f), (screenMinY + screenMaxY) / 2 };

    // ray casting helpers
    ScreenRayCasting scrc = makeScreenRayCasting(view, front, up, rightVec, screenW, screenH, fov);
    glm::vec3 dx = scrc.dx;
    glm::vec3 dy = scrc.dy;
    glm::vec3 rayStart = scrc.rayStart;

    float pixelDepths[5];
    for (int i = 0; i < 5; ++i) {
        glm::vec3 rd = computeRd(xcoords[i], ycoords[i], screenW, screenH, rightVec, up, front, fov);
        pixelDepths[i] = onSphDepth(rd, xcoords[i], ycoords[i], glm::vec3(posr), posr.w, rayStart, dx, dy, proj, cameraPos);
    }

    bool billboardVisible = isSphereBillboardVisible(downsampleTex, screenW, screenH, xcoords, ycoords, pixelDepths, 5);

    unsigned int visIdx = visibilityFrameBufferIndexOffset + idx;
    if (!billboardVisible) {
        if (visibilityFrameBuffer[visIdx].frames > 0 && visibilityFrameBuffer[visIdx].pixels == 0) {
            visibilityFrameBuffer[visIdx].frames -= 1;
        } else if (visibilityFrameBuffer[visIdx].frames == 0 && visibilityFrameBuffer[visIdx].pixels == 0) {
            // fully occluded, bail out
            return;
        } else {
            visibilityFrameBuffer[visIdx].frames = 10;
        }
    } else {
        visibilityFrameBuffer[visIdx].frames = 10;
    }

    if (benchmark == 1) atomicAdd(occCullcounter, 1u);

    // rasterize inside bbox
    int minx = max(0, screenMinX);
    int maxx = min(screenMaxX, screenW);
    int miny = max(0, screenMinY);
    int maxy = min(screenMaxY, screenH);

    for (int px = minx; px < maxx; ++px) {
        glm::vec3 rayColStart = rayStart + float(px) * dx;
        bool finishedLine = false;
        for (int py = miny; py < maxy; ++py) {
            // surf2Dwrite(make_uchar4(255, 255, 255, 255), outputImage,
            // static_cast<unsigned int>(px) * sizeof(uchar4),
            // static_cast<unsigned int>(py));
            glm::vec3 rd = computeRd(px, py, screenW, screenH, rightVec, up, front, fov);
            float t = iSphere(glm::vec3(cameraPos), rd, glm::vec3(posr), posr.w);
            if (t > 0.0f) {
                finishedLine = true;
                glm::vec3 hit = (rayColStart + float(py) * dy) * t;
                float proj2z = proj[2][2];
                float proj3z = proj[3][2];
                float depth = (hit.z * proj2z + proj3z) / -hit.z;

                unsigned int depthU = floatToUintBits(depth);
                int index = px + screenW * py;

                // atomicMin over uint bits (we store depth as float bits in uint)
                unsigned int old = atomicMin(&depthBuffer[index], depthU);

                // if we won (old > depth)
                if (uintToFloatBits(old) > depth) {
                    // own the pixel
                    pixelOwnershipBuffer[index] = (unsigned int)idx;
                    glm::vec3 normal = glm::normalize(glm::vec3(cameraPos) + rd * t - glm::vec3(posr));
                    float lambert = glm::max(0.0f, glm::dot(normal, -glm::normalize(rd * t)));
                    glm::vec3 color = glm::vec3(0.01f, 1.0f, 0.05f) * lambert * 0.9f;
                    uchar4 ucharColor;
                    ucharColor = make_uchar4(color.x*255, color.y*255, color.z*255, 255);
                    surf2Dwrite(ucharColor, outputImage, px * sizeof(uchar4), py); 
                }
            } else if (finishedLine) {
                break;
            }
        }
    }

    // clear pixel counter like shader
    visibilityFrameBuffer[visIdx].pixels = 0;
}


extern "C" void sphereRaster(
    glm::vec4* d_spheres,
    int sphereCount,
    glm::mat4 view,
    glm::mat4 proj,
    glm::vec4 frustumTopFace,
    glm::vec4 frustumBottomFace,
    glm::vec4 frustumRightFace,
    glm::vec4 frustumLeftFace,
    glm::vec4 frustumFarFace,
    glm::vec4 frustumNearFace,
    glm::vec3 front,
    glm::vec3 up,
    glm::vec3 rightVec,
    glm::vec3 cameraPos,
    int screenW,
    int screenH,
    float fov,
    unsigned int* d_depthBuffer,
    unsigned int* d_pixelOwnershipBuffer,
    unsigned int* d_visibilityFrameBuffer,
    unsigned int visibilityFrameBufferIndexOffset,
    unsigned int* d_frustCullcounter,
    unsigned int* d_occCullcounter,
    int benchmark,
    cudaSurfaceObject_t outputImage,
    cudaTextureObject_t downsampleTex
)
{
    dim3 block(128);
    dim3 grid((sphereCount + block.x - 1) / block.x);
    sphereRasterKernel<<<grid, block>>>(
        d_spheres,
        sphereCount,
        view,
        proj,
        frustumTopFace,
        frustumBottomFace,
        frustumRightFace,
        frustumLeftFace,
        frustumFarFace,
        frustumNearFace,
        front,
        up,
        rightVec,
        cameraPos,
        screenW,
        screenH,
        fov,
        d_depthBuffer,
        d_pixelOwnershipBuffer,
        (FramePixelsCount*)d_visibilityFrameBuffer,
        visibilityFrameBufferIndexOffset,
        d_frustCullcounter,
        d_occCullcounter,
        benchmark,
        outputImage,
        downsampleTex
    );
    cudaDeviceSynchronize();
}