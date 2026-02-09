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
#include "utils/scene_descriptor.h"

#define pixelsToCheck 3

// ========================================================
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
    int cylinderCount;
    unsigned int visibilityFrameBufferIndexOffset;
    int benchmark;
};

static __constant__ Constants cst;

struct CylinderIndex
{
    unsigned int sphereIndexA;
    unsigned int sphereIndexB;
    float radius;
    unsigned int _padding;
};
using Cylinder = CylinderIndex;

struct BBox3D
{
    glm::vec3 mMin;
    glm::vec3 mMax;
};

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

__device__ inline glm::vec2 safeMin(const glm::vec2& a, const glm::vec2& b) {
    return glm::vec2(fminf(a.x, b.x), fminf(a.y, b.y));
}

__device__ inline glm::vec2 safeMax(const glm::vec2& a, const glm::vec2& b) {
    return glm::vec2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
}

__device__ inline glm::vec3 safeMin(const glm::vec3& a, const glm::vec3& b) {
    return glm::vec3(
        fminf(a.x, b.x),
        fminf(a.y, b.y),
        fminf(a.z, b.z)
    );
}

__device__ inline glm::vec3 safeMax(const glm::vec3& a, const glm::vec3& b) {
    return glm::vec3(
        fmaxf(a.x, b.x),
        fmaxf(a.y, b.y),
        fmaxf(a.z, b.z)
    );
}

static __device__ inline glm::vec4 iCylinder(const glm::vec3& ro, const glm::vec3& rd, const glm::vec3& pa, const glm::vec3& pb, float ra) {
    glm::vec3  ba = pb - pa;
    glm::vec3  oc = ro - pa;

    float baba = glm::dot(ba,ba);
    float bard = glm::dot(ba,rd);
    float baoc = glm::dot(ba,oc);
    
    float k2 = fmaf(bard, -bard, baba);
    float k1 = fmaf(glm::dot(oc,rd), baba, -baoc*bard);
    float k0 = fmaf(baba, glm::dot(oc,oc), fmaf(- ra*ra, baba, -baoc*baoc));
    
    
    float h = fmaf(k1, k1, -k2*k0);
    if( h<0.0f ) return glm::vec4(-1.0f);
    h = sqrt(h);
    float t = __fdividef(-k1-h, k2);

    // body
    float y = fmaf(t, bard, baoc);
    if( y>0.0f && y<baba ) return glm::vec4( t, oc+t*rd - ba*y/baba );
    
    // caps
    // t = ( ((y<0.0f) ? 0.0f : baba) - baoc)/bard;
    // if( abs(k1+k2*t)<h ) return vec4( t, ba*sign(y)/sqrt(baba) );

    return glm::vec4(-1.0f);
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
__device__ inline float texelFetchDepth(cudaTextureObject_t tex, int x, int y) {
    // use integer coords + 0.5f to sample exact texel (readMode = cudaReadModeElementType)
    return tex2D<float>(tex, float(x) + 0.5f, float(y) + 0.5f);
}


__device__ inline float intersectX(glm::vec2 a, glm::vec2 b, float y) 
{
    if (a.y == b.y) return a.x; // horizontal line
    float t = __fdividef(y - a.y, b.y - a.y);
    return fmaf(t, (b.x - a.x), a.x);
}

// billboard pre-test: samples 5 texels from downsampled texture
__device__ inline bool isCylinderBillboardVisible(
    cudaTextureObject_t downsampleTex,
    int xcoords[], int ycoords[], float pixelDepths[])
{
    int dsW = max(1, cst.screenW / 16);
    int dsH = max(1, cst.screenH / 16);

    // sample mapped coords (like original)
    float downvals[3];
    float dsW_scrW = __fdividef(float(dsW), float(cst.screenW));
    float dsH_scrH = __fdividef(float(dsH), float(cst.screenH));

    #pragma unroll pixelsToCheck
    for (int i = 0; i < pixelsToCheck; ++i) {
        int tx = int(float(xcoords[i]) * dsW_scrW);
        int ty = int(float(ycoords[i]) * dsH_scrH);
        tx = clampi(tx, 0, dsW - 1);
        ty = clampi(ty, 0, dsH - 1);
        downvals[i] = texelFetchDepth(downsampleTex, tx, ty);
    }

    #pragma unroll pixelsToCheck
    for (int i = 0; i < pixelsToCheck; ++i) {
        float pd = pixelDepths[i];
        if (pd <= downvals[0] || pd <= downvals[1] || pd <= downvals[2])
            return true;
    }
    return false;
}

__device__ inline bool isOnOrForwardPlaneAABB(glm::vec4 plane, BBox3D& bbox) 
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

__device__ inline bool isCylinderInside(const glm::vec3 pa,
                                        const glm::vec3 pb,
                                        float radius)
{
    glm::vec3 a = pb - pa;
    glm::vec3 e = radius*sqrt( 1.0f - a*a/glm::dot(a,a) );
    BBox3D bbox3d = BBox3D{safeMin( pa - e, pb - e ), safeMax( pa + e, pb + e )};
    return isOnOrForwardPlaneAABB(cst.frustumPlanes[0], bbox3d) &&
           isOnOrForwardPlaneAABB(cst.frustumPlanes[1], bbox3d) &&
           isOnOrForwardPlaneAABB(cst.frustumPlanes[2], bbox3d) &&
           isOnOrForwardPlaneAABB(cst.frustumPlanes[3], bbox3d) &&
           isOnOrForwardPlaneAABB(cst.frustumPlanes[4], bbox3d) &&
           isOnOrForwardPlaneAABB(cst.frustumPlanes[5], bbox3d);
}

// compute eye ray direction (equivalent to computeRd in GLSL)
__device__ inline glm::vec3 computeRd(int px, int py, const float& fovTan, const float& halfFovTan)
{
    glm::vec2 p = ( -glm::vec2(cst.screenW, cst.screenH) + 2.0f * glm::vec2(px, py) ) / glm::vec2(cst.screenW, cst.screenH);
    return glm::normalize(p.x * cst.right * halfFovTan + p.y * cst.up * fovTan + cst.front);
}

// onSphDepth same math as GLSL
__device__ inline float onCylDepth(const glm::vec3& rd, int px, int py, const glm::vec3& pa, const glm::vec3& pb, float r,
                            const glm::vec3& rayStart, const glm::vec3& dx, const glm::vec3& dy)
{
    float h = iCylinder(cst.cameraPos, rd, pa, pb, r).x;
    glm::vec3 hit = (rayStart + float(px) * dx + float(py) * dy) * h;
    // project depth like GLSL: (hit.z * proj[2].z + proj[3].z) / -hit.z

    return __fdividef(fmaf(hit.z, cst.proj[2][2], cst.proj[3][2]), -hit.z);
}

struct RayDirectionResult 
{
    glm::vec3 rd;
    glm::ivec2 screenPos;
};

__device__ inline RayDirectionResult getRayDirection(glm::vec3& pa, glm::vec3& pb, const float& fovTan, const float& halfFovTan)
{
    glm::vec3 p = pa.z < pb.z ? pa : pb;
    const glm::vec4 pProj = cst.proj * glm::vec4(p, 1.0f);
    p = glm::vec3(__fdividef(pProj.x, pProj.w), __fdividef(pProj.y, pProj.w), __fdividef(pProj.z, pProj.w));
    const glm::ivec2 pScreen = glm::ivec2(int(fmaf(p.x, 0.5f, 0.5f)) * cst.screenW, int(fmaf(p.y, 0.5f, 0.5f)) * cst.screenH);
    return {computeRd(pScreen.x, pScreen.y, fovTan, halfFovTan), pScreen};
}

// ========================================================
// Kernel: one thread per sphere
// ========================================================
__global__ void cylinderRasterKernel(
    // buffers (device pointers)
    Cylinder* __restrict__ cylinders,
    const glm::vec4* __restrict__ spheres,
    // storage buffers (lineal)
    unsigned int* __restrict__ depthBuffer,            // in uint bits of float depth, length = screenW*screenH
    unsigned int* __restrict__ pixelOwnershipBuffer,   // length = screenW*screenH
    FramePixelsCount* __restrict__ visibilityFrameBuffer,
    // counters
    unsigned int*  frustCullcounter,
    unsigned int*  occCullcounter,
    // outputs (linear) -- optional, can pass nullptr if unused
    cudaSurfaceObject_t outputImage,         // RGBA8 linear buffer [screenW*screenH*4]
    // downsample texture object for billboard test
    cudaTextureObject_t downsampleTex
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= cst.cylinderCount) return;

    Cylinder c = cylinders[idx];
    glm::vec4 sA = spheres[c.sphereIndexA];
    glm::vec4 sB = spheres[c.sphereIndexB];
    glm::vec3 pa = glm::vec3(sA);
    glm::vec3 pb = glm::vec3(sB);
    float ra = c.radius;

    // Frustum test
    if (!isCylinderInside(pa, pb, ra))
    {
        // mark zero pixels like shader did
        unsigned int visId = cst.visibilityFrameBufferIndexOffset + idx;
        visibilityFrameBuffer[visId].pixels = 0;

        return;
    }

    if (cst.benchmark == 1) atomicAdd(frustCullcounter, 1u);
    glm::vec3 camSpaceCylA = glm::vec3(cst.view * glm::vec4(pa, 1.0f));
    glm::vec3 camSpaceCylB = glm::vec3(cst.view * glm::vec4(pb, 1.0f));

    glm::vec3 camImpPosA, camImpPosB;
    if ( camSpaceCylA.z < camSpaceCylB.z )
	{
		camImpPosA = camSpaceCylB;
		camImpPosB = camSpaceCylA;
	}
	else
	{
		camImpPosA = camSpaceCylA;
		camImpPosB = camSpaceCylB;
	}
    glm::vec3 center = normalize( ( camImpPosA + camImpPosB ) * 0.5f );
    glm::vec2 projectedPoints[4];
    // Cylinder axis
    const glm::vec3 z = normalize(camImpPosB - camImpPosA);

    // Find orthonormal x,y axes orthogonal to cylinder axis
    glm::vec3 x = normalize(cross(center, z));
    glm::vec3 y = normalize(cross(x, z)); // make full basis

    // Compute impostor construction vectors.
    const float dV0 = length( camImpPosA );
    const float dV1 = length( camImpPosB );

    const float sinAngle = __fdividef(ra, dV0);
    float		angle	 = asinf( sinAngle );
    const glm::vec3	y1		 = y * ra;
    const glm::vec3	x2		 = x * ra * cosf( angle );
    const glm::vec3	y2		 = y1 * sinAngle;
    angle				 = asinf( __fdividef(ra, dV1) );
    const glm::vec3 x3		 = x * ( dV1 - ra ) * tanf( angle );

    // Compute impostors vertices.
    const glm::vec3 v1 = camImpPosA - x2 + y2;
    const glm::vec3 v2 = camImpPosA + x2 + y2;
    const glm::vec3 v3 = camImpPosB - x3 + y1;
    const glm::vec3 v4 = camImpPosB + x3 + y1;

    const glm::vec4 v1Proj = cst.proj * glm::vec4(v1, 1.0f);
    const glm::vec4 v2Proj = cst.proj * glm::vec4(v2, 1.0f);
    const glm::vec4 v3Proj = cst.proj * glm::vec4(v3, 1.0f);
    const glm::vec4 v4Proj = cst.proj * glm::vec4(v4, 1.0f);

    glm::vec3 ndcv1Proj = glm::vec3(__fdividef(v1Proj.x, v1Proj.w), __fdividef(v1Proj.y, v1Proj.w), __fdividef(v1Proj.z, v1Proj.w));
    glm::vec3 ndcv2Proj = glm::vec3(__fdividef(v2Proj.x, v2Proj.w), __fdividef(v2Proj.y, v2Proj.w), __fdividef(v2Proj.z, v2Proj.w));
    glm::vec3 ndcv3Proj = glm::vec3(__fdividef(v3Proj.x, v3Proj.w), __fdividef(v3Proj.y, v3Proj.w), __fdividef(v3Proj.z, v3Proj.w));
    glm::vec3 ndcv4Proj = glm::vec3(__fdividef(v4Proj.x, v4Proj.w), __fdividef(v4Proj.y, v4Proj.w), __fdividef(v4Proj.z, v4Proj.w));

    projectedPoints[0] = glm::vec2(ndcv1Proj);
    projectedPoints[1] = glm::vec2(ndcv2Proj);
    projectedPoints[2] = glm::vec2(ndcv4Proj);
    projectedPoints[3] = glm::vec2(ndcv3Proj);

    int minX = cst.screenW;
    int maxX = 0;
    int minY = cst.screenH;
    int maxY = 0;
    #pragma unroll 4
    for (int i = 0; i < 4; i++) 
    {
        projectedPoints[i].x = int(fmaf(projectedPoints[i].x, 0.5f, 0.5f) * cst.screenW);
        projectedPoints[i].y = int(fmaf(projectedPoints[i].y, 0.5f, 0.5f) * cst.screenH);

        minY = min(minY, __float2int_rd(projectedPoints[i].y)); 
        maxY = max(maxY, __float2int_ru(projectedPoints[i].y));
        minX = min(minX, __float2int_rd(projectedPoints[i].x)); 
        maxX = max(maxX, __float2int_ru(projectedPoints[i].x));
    }

    if (maxY < 0 || minY >= cst.screenH || maxY - minY < 2 ||
        maxX < 0 || minX >= cst.screenW || maxX - minX < 2) {
        // outside screen or too small
        unsigned int visId = cst.visibilityFrameBufferIndexOffset + idx;
        visibilityFrameBuffer[visId].pixels = 0;
        return;
    }

    float aspectRatio = __fdividef(float(cst.screenW), float(cst.screenH));
    float fovRad = glm::radians(cst.fov);
    float fovTan = tanf(fovRad * 0.5f);
    float halfFovTan = fovTan * aspectRatio;


    // ray casting helpers
    ScreenRayCasting scrc = makeScreenRayCasting(fovRad, fovTan, halfFovTan);
    glm::vec3 dx = scrc.dx;
    glm::vec3 dy = scrc.dy;
    glm::vec3 rayStart = scrc.rayStart;

    glm::vec3 closePa1 = camImpPosA - x * ra;
    glm::vec3 closePa2 = camImpPosA + x * ra;
    RayDirectionResult closePa = getRayDirection(closePa1, closePa2, fovTan, halfFovTan);
    
    glm::vec3 closePb1 = camImpPosB - x * ra;
    glm::vec3 closePb2 = camImpPosB + x * ra;
    RayDirectionResult closePb = getRayDirection(closePb1, closePb2, fovTan, halfFovTan);
    
    glm::vec3 closeCenter1 = center - x * ra;
    glm::vec3 closeCenter2 = center + x * ra;
    RayDirectionResult closeCenter = getRayDirection(closeCenter1, closeCenter2, fovTan, halfFovTan);
    int xcoords[3] = {closePa.screenPos.x, closePb.screenPos.x, closeCenter.screenPos.x};
    int ycoords[3] = {closePa.screenPos.y, closePb.screenPos.y, closeCenter.screenPos.y};

    float pixelDepths[3] = {
        onCylDepth(closePa.rd, closePa.screenPos.x, closePa.screenPos.y, pa, pb, ra, rayStart, dx, dy),
        onCylDepth(closePb.rd, closePb.screenPos.x, closePb.screenPos.y, pa, pb, ra, rayStart, dx, dy),
        onCylDepth(closeCenter.rd, closeCenter.screenPos.x, closeCenter.screenPos.y, pa, pb, ra, rayStart, dx, dy)
    };

    bool billboardVisible = isCylinderBillboardVisible(downsampleTex, xcoords, ycoords, pixelDepths);

    unsigned int visIdx = cst.visibilityFrameBufferIndexOffset + idx;
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

    if (cst.benchmark == 1) atomicAdd(occCullcounter, 1u);

    for (int py = fminf(cst.screenH-1, maxY); py >= fmaxf(0, minY); --py)
    {
        float xIntersections[4];
        int intersections = 0;

        #pragma unroll 4
        for (int i = 0; i < 4; ++i) 
        {
            const glm::vec2 a = projectedPoints[i];
            const glm::vec2 b = projectedPoints[(i + 1) % 4];

            if ((py >= a.y && py <= b.y) || (py >= b.y && py <= a.y)) 
            {
                float x = intersectX(a, b, py);
                xIntersections[intersections] = x;
                intersections++;
            }
        }

        if (intersections >= 2)
        {
            float xMin = xIntersections[0];
            float xMax = xIntersections[0];

            for (int i = 1; i < intersections; ++i) 
            {
                xMin = fminf(xMin, xIntersections[i]);
                xMax = fmaxf(xMax, xIntersections[i]);
            }

            glm::vec3 rayColStart = rayStart + float(py) * dy;

            for (int px = fmaxf(0, __float2int_ru(xMin)); px <= fminf(__float2int_rd(xMax), cst.screenW -1); ++px)
            {
                glm::vec3 rd = computeRd(px, py, fovTan, halfFovTan);
                glm::vec4 tnor = iCylinder(glm::vec3(cst.cameraPos), rd, pa, pb, ra);
                int index = px + cst.screenW*py;

                if (tnor.x > 0.0f) {
                    float t = tnor.x;
                    glm::vec3 hit = (rayColStart + float(px) * dx) * t;
                    float depth = (__fdividef(fmaf(hit.z, cst.proj[2][2], cst.proj[3][2]), -hit.z));

                    unsigned int depthU = floatToUintBits(depth);
                    
                    // atomicMin over uint bits (we store depth as float bits in uint)
                    unsigned int old = atomicMin(&depthBuffer[index], depthU);

                    // if we won (old > depth)
                    if (uintToFloatBits(old) > depth) {
                        // own the pixel
                        pixelOwnershipBuffer[index] = (unsigned int)visIdx;
                        glm::vec3 normal = glm::normalize(glm::vec3(tnor.y, tnor.z, tnor.w));
                        float lambert = glm::max(0.0f, glm::dot(normal, -glm::normalize(rd * t)));
                        glm::vec3 color = glm::vec3(bondsColor.x, bondsColor.y, bondsColor.z) * lambert * diffuse;
                        uchar4 ucharColor = make_uchar4(color.x*255, color.y*255, color.z*255, 255);
                        surf2Dwrite(ucharColor, outputImage, px * sizeof(uchar4), py); 
                    }
                } 
            }
        }
    }

    // clear pixel counter like shader
    visibilityFrameBuffer[visIdx].pixels = 0;
}


extern "C" void cylinderRaster(
    Cylinder* cylinders,
    const glm::vec4* d_spheres,
    int c_cylinderCount,
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
    unsigned int* d_pixelOwnershipBuffer,
    unsigned int* d_visibilityFrameBuffer,
    unsigned int c_visibilityFrameBufferIndexOffset,
    unsigned int* d_frustCullcounter,
    unsigned int* d_occCullcounter,
    int c_benchmark,
    cudaSurfaceObject_t outputImage,
    cudaTextureObject_t downsampleTex,
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
    c_cylinderCount,
    c_visibilityFrameBufferIndexOffset,
    c_benchmark
    };
    // copy constants to symbol memory (per-frame)
    cudaMemcpyToSymbolAsync(cst, &c_constants, sizeof(Constants), 0, cudaMemcpyHostToDevice, stream);

    nvtxRangePushA("Cylinder Raster Kernel");
    dim3 block(workGroupSizeX);
    dim3 grid((c_cylinderCount + block.x - 1) / block.x);
    cylinderRasterKernel<<<grid, block, 0, stream>>>(
        cylinders,
        d_spheres,
        d_depthBuffer,
        d_pixelOwnershipBuffer,
        (FramePixelsCount*)d_visibilityFrameBuffer,
        d_frustCullcounter,
        d_occCullcounter,
        outputImage,
        downsampleTex
    );
    // cudaDeviceSynchronize();
    nvtxRangePop();
}