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

__constant__ float atomsColor[3] = {0.8f, 0.1f, 0.1f};
__constant__ float bondsColor[3] = {0.2f, 0.5f, 0.9f};
__constant__ float diffuse = 0.9f;

__device__ inline float intersectX(glm::vec2 a, glm::vec2 b, float y) 
{
    if (a.y == b.y) return a.x; // horizontal line
    float t = __fdividef(y - a.y, b.y - a.y);
    return fmaf(t, (b.x - a.x), a.x);
}

__device__ inline float iSphere(const glm::vec3& ro, const glm::vec3& rd,
                                const glm::vec3& center, float radius) {
    glm::vec3 oc = ro - center;
    float b = glm::dot(oc, rd);
    float c = glm::dot(oc, oc) - radius * radius;
    float h = b * b - c;
    if (h < 0.0f) return -1.0f;
    return -b - sqrtf(h);
}

__device__ inline glm::vec4 iCylinder(const glm::vec3& ro, const glm::vec3& rd,
                                      const glm::vec3& pa, const glm::vec3& pb, float ra) {
    glm::vec3 ba = pb - pa, oc = ro - pa;
    float baba = glm::dot(ba, ba), bard = glm::dot(ba, rd), baoc = glm::dot(ba, oc);
    
    float k2 = fmaf(bard, -bard, baba);
    float k1 = fmaf(glm::dot(oc,rd), baba, -baoc*bard);
    float k0 = fmaf(baba, glm::dot(oc,oc), fmaf(- ra*ra, baba, -baoc*baoc));
    
    float h = fmaf(k1, k1, -k2*k0);
    if (h < 0.0f) return glm::vec4(-1.0f);

    h = sqrtf(h);
    float t =  __fdividef(-k1-h, k2);

    // body
    float y = fmaf(t, bard, baoc);
    if( y>0.0f && y<baba ) return glm::vec4( t, oc+t*rd - ba*y/baba );
    
    // caps
    // t = ( ((y<0.0f) ? 0.0f : baba) - baoc)/bard;
    // if( abs(k1+k2*t)<h ) return vec4( t, ba*sign(y)/sqrt(baba) );

    return glm::vec4(-1.0f);
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

    int screenMinX = max(0, __float2int_rd(fmaf(minC.x, 0.5f, 0.5f) * hybridCst.screenWidth));
    int screenMinY = max(0, __float2int_rd(fmaf(minC.y, 0.5f, 0.5f) * hybridCst.screenHeight));
    int screenMaxX = min(hybridCst.screenWidth,  __float2int_ru(fmaf(maxC.x, 0.5f, 0.5f) * hybridCst.screenWidth));
    int screenMaxY = min(hybridCst.screenHeight, __float2int_ru(fmaf(maxC.y, 0.5f, 0.5f) * hybridCst.screenHeight));


    float difx = float(screenMaxX - screenMinX);
    float dify = float(screenMaxY - screenMinY);
    if (difx * dify <= 2.0f) return;

    float aspectRatio = __fdividef(float(hybridCst.screenWidth), float(hybridCst.screenHeight));
    float fovRad = glm::radians(hybridCst.fov);
    float fovTan = __tanf(fovRad * 0.5f);
    float halfFovTan = fovTan * aspectRatio;


    float proj22 = hybridCst.proj[2][2];
    float proj32 = hybridCst.proj[3][2];

    for (int py = screenMinY; py < screenMaxY; py++) {
        glm::vec3 rayColStart = hybridCst.rayStart + (float)py * hybridCst.dy;
        for (int px = screenMinX; px < screenMaxX; px++) {
            glm::vec3 rd = computeRayDirection(px, py, fovTan, halfFovTan);
            float t = iSphere(hybridCst.cameraPos, rd, spherePos, radius);

            if (t > 0.0f) {
                glm::vec3 hit = (rayColStart + (float)px * hybridCst.dx) * t;
                float depth = __fdividef(fmaf(hit.z, proj22, proj32), -hit.z);
                unsigned int depthU = __float_as_uint(depth);

                int pixelIdx = py * hybridCst.screenWidth + px;
                unsigned int old = atomicMin(&depthBuffer[pixelIdx], depthU);
                if (__uint_as_float(old) > depth) {
                    glm::vec3 normal = glm::normalize(hybridCst.cameraPos + rd * t - glm::vec3(spherePosR));
                    float lambert = fmaxf(0.0f, glm::dot(normal, -glm::normalize(rd * t)));
                    glm::vec3 color = glm::vec3(atomsColor[0], atomsColor[1], atomsColor[2]) * lambert * diffuse;
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
    glm::vec2 projectedPoints[4];
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

    projectedPoints[0] = glm::vec2(ndcv1Proj);
    projectedPoints[1] = glm::vec2(ndcv2Proj);
    projectedPoints[2] = glm::vec2(ndcv4Proj);
    projectedPoints[3] = glm::vec2(ndcv3Proj);

    int minX = hybridCst.screenWidth;
    int maxX = 0;
    int minY = hybridCst.screenHeight;
    int maxY = 0;

    #pragma unroll 4
    for (int i = 0; i < 4; i++) 
    {
        projectedPoints[i].x = int(fmaf(projectedPoints[i].x, 0.5f, 0.5f) * hybridCst.screenWidth);
        projectedPoints[i].y = int(fmaf(projectedPoints[i].y, 0.5f, 0.5f) * hybridCst.screenHeight);

        minY = min(minY, __float2int_rd(projectedPoints[i].y)); 
        maxY = max(maxY, __float2int_ru(projectedPoints[i].y));
        minX = min(minX, __float2int_rd(projectedPoints[i].x)); 
        maxX = max(maxX, __float2int_ru(projectedPoints[i].x));
    }

    if (maxY < 0 || minY >= hybridCst.screenHeight || maxY - minY < 2 ||
        maxX < 0 || minX >= hybridCst.screenWidth || maxX - minX < 2) return;
    

    float aspectRatio = fdividef(hybridCst.screenWidth, hybridCst.screenHeight);
    float fovRad = glm::radians(hybridCst.fov);
    float fovTan = __tanf(fovRad * 0.5f);
    float halfFovTan = fovTan * aspectRatio;

    for (int py = fminf(hybridCst.screenHeight-1, maxY); py >= fmaxf(0, minY); --py)
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
            glm::vec3 rayColStart = hybridCst.rayStart + (float)py * hybridCst.dy;
            for (int px = fmaxf(0, __float2int_ru(xMin)); px <= fminf(__float2int_rd(xMax), hybridCst.screenWidth -1); ++px)
            {
                glm::vec3 rd = computeRayDirection(px, py, fovTan, halfFovTan);
                glm::vec4 tnor = iCylinder(glm::vec3(hybridCst.cameraPos), rd, pa, pb, radius);
                int index = px + hybridCst.screenWidth*py;

                if (tnor.x > 0.0f) {
                    float t = tnor.x;
                    glm::vec3 hit = (rayColStart + (float)px * hybridCst.dx) * t;
                    float depth = __fdividef(fmaf(hit.z, hybridCst.proj[2][2], hybridCst.proj[3][2]), -hit.z);

                    unsigned int depthU = __float_as_uint(depth);
                    
                    // atomicMin over uint bits (we store depth as float bits in uint)
                    unsigned int old = atomicMin(&depthBuffer[index], depthU);

                    // if we won (old > depth)
                    if (__uint_as_float(old) > depth) {
                        // own the pixel
                        glm::vec3 normal = glm::normalize(glm::vec3(tnor.y, tnor.z, tnor.w));
                        float lambert = glm::max(0.0f, glm::dot(normal, -glm::normalize(rd * t)));
                        glm::vec3 color = glm::vec3(bondsColor[0], bondsColor[1], bondsColor[2]) * lambert * diffuse;
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
