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
    if( y>0.0f && y<baba ) 
        return glm::vec4( t, oc+t*rd - ba*y/baba );
    
    // caps
    // t = ( ((y<0.0f) ? 0.0f : baba) - baoc)/bard;
    // if( abs(k1+k2*t)<h ) return vec4( t, ba*sign(y)/sqrt(baba) );

    return glm::vec4(-1.0f);
}

__device__ inline glm::vec3 fastNormalize(const glm::vec3& v) {
    return v * rsqrtf(glm::dot(v, v));
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

    unsigned int sphereIdx = __ldg(&smallSphereIndices[idx]);

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

    const float proj22 = hybridCst.proj[2][2];
    const float proj32 = hybridCst.proj[3][2];

    const int difx = screenMaxX - screenMinX;
    const int dify = screenMaxY - screenMinY;

    // ---- Point fallback for sub-pixel spheres ---------------------------
    // Single-ray-per-pixel sampling produces concentric-ring moire when the
    // sphere is smaller than a pixel: the ray may miss the sphere depending
    // on sub-pixel alignment. For tiny bboxes we draw the projected center
    // pixel directly with the sphere's center depth.
    if (difx <= 1 && dify <= 1) {
        glm::vec4 centerClip = hybridCst.proj * camSpace4;
        if (centerClip.w > 1e-6f) {
            float invW = __fdividef(1.0f, centerClip.w);
            float ndcX = centerClip.x * invW;
            float ndcY = centerClip.y * invW;
            float ndcZ = centerClip.z * invW;
            int px = __float2int_rd(fmaf(ndcX, 0.5f, 0.5f) * hybridCst.screenWidth);
            int py = __float2int_rd(fmaf(ndcY, 0.5f, 0.5f) * hybridCst.screenHeight);
            if (px >= 0 && px < hybridCst.screenWidth &&
                py >= 0 && py < hybridCst.screenHeight) {
                int pixelIdx = py * hybridCst.screenWidth + px;
                unsigned int depthU = __float_as_uint(ndcZ);
                unsigned int old = atomicMin(&depthBuffer[pixelIdx], depthU);
                if (__uint_as_float(old) > ndcZ) {
                    // Constant mid-range lambert keeps far points visually
                    // consistent with shaded near spheres.
                    glm::vec3 color = glm::vec3(atomsColor[0], atomsColor[1], atomsColor[2]) * 0.7f * diffuse;
                    uchar4 ucharColor = make_uchar4(
                        (unsigned char)(color.x * 255.0f),
                        (unsigned char)(color.y * 255.0f),
                        (unsigned char)(color.z * 255.0f), 255);
                    surf2Dwrite(ucharColor, outputImage, px * sizeof(uchar4), py);
                }
            }
        }
        return;
    }

    // ---- Radius clamp for multi-pixel small spheres ---------------------
    // Bumps the effective ray-test radius so that the closest pixel center
    // (worst case sqrt(2)/2 px away) is guaranteed to hit. Without this,
    // 2x2 / 2x3 / 3x2 bboxes still produce moire holes.
    float pxSize = dist * hybridCst.pxScale;
    float effRadius = fmaxf(radius, 0.7071f * pxSize);

    for (int py = screenMinY; py < screenMaxY; py++) {
        glm::vec3 rayColStart = hybridCst.rayStart + (float)py * hybridCst.dy;
        for (int px = screenMinX; px < screenMaxX; px++) {
            glm::vec3 ray = rayColStart + (float)px * hybridCst.dx;
            glm::vec3 rd = fastNormalize(ray.x * hybridCst.right + ray.y * hybridCst.up - ray.z * hybridCst.front);
            float t = iSphere(hybridCst.cameraPos, rd, spherePos, effRadius);

            if (t > 0.0f) {
                glm::vec3 hit = ray * t;
                float depth = __fdividef(fmaf(hit.z, proj22, proj32), -hit.z);
                unsigned int depthU = __float_as_uint(depth);

                int pixelIdx = py * hybridCst.screenWidth + px;
                unsigned int old = atomicMin(&depthBuffer[pixelIdx], depthU);
                if (__uint_as_float(old) > depth) {
                    glm::vec3 normal = fastNormalize(hybridCst.cameraPos + rd * t - glm::vec3(spherePosR));
                    float lambert = fmaxf(0.0f, glm::dot(normal, -rd));
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
    const glm::vec4* __restrict__ spheres,
    const unsigned int* __restrict__ smallCylinderIndices,
    unsigned int smallCylinderCount,
    unsigned int* __restrict__ depthBuffer,
    cudaSurfaceObject_t outputImage)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= smallCylinderCount) return;
    unsigned int cylIdx = __ldg(&smallCylinderIndices[idx]);
    Cylinder cyl = cylinders[cylIdx];
    glm::vec4 sA = spheres[cyl.sphereIndexA];
    glm::vec4 sB = spheres[cyl.sphereIndexB];
    glm::vec3 pa = glm::vec3(sA), pb = glm::vec3(sB);
    float radius = cyl.radius;
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
    const glm::vec3	x2		 = x * radius * __cosf( angle );
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

    // Defense in depth: with the new classifier this should never trigger,
    // but if any impostor vertex is at/behind the near plane the perspective
    // divide produces sign-flipped NDC, which previously made one thread
    // sweep the whole screen and triggered WDDM preemption.
    if (v1Proj.w <= 1e-3f || v2Proj.w <= 1e-3f ||
        v3Proj.w <= 1e-3f || v4Proj.w <= 1e-3f) {
        return;
    }

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

    // Hard cap: a "small" cylinder must fit comfortably inside a few times
    // the configured threshold. If it doesn't, the classifier produced a
    // false positive (legacy data, race, or stale h_smallCount); bail out
    // so this thread cannot block the whole GPU.
    {
        const int clampedMinX = max(0, minX);
        const int clampedMinY = max(0, minY);
        const int clampedMaxX = min(hybridCst.screenWidth,  maxX);
        const int clampedMaxY = min(hybridCst.screenHeight, maxY);
        const int bboxW = clampedMaxX - clampedMinX;
        const int bboxH = clampedMaxY - clampedMinY;
        if (bboxW <= 0 || bboxH <= 0) return;
        const long long bboxArea = (long long)bboxW * (long long)bboxH;
        const long long maxSmallArea = (long long)hybridCst.smallEntityThreshold * 4LL;
        if (bboxArea > maxSmallArea) return;
    }

    // ---- Point fallback for sub-pixel cylinders -----------------------------
    // Avoid concentric-ring moire from sub-pixel ray-cylinder tests by writing
    // a single pixel at the projected midpoint when the impostor footprint is
    // <=1x1 pixel.
    if ((maxX - minX) <= 1 && (maxY - minY) <= 1) {
        glm::vec3 midWorld = (pa + pb) * 0.5f;
        glm::vec4 midClip = hybridCst.proj * hybridCst.view * glm::vec4(midWorld, 1.0f);
        if (midClip.w > 1e-6f) {
            float invW = __fdividef(1.0f, midClip.w);
            float ndcX = midClip.x * invW;
            float ndcY = midClip.y * invW;
            float ndcZ = midClip.z * invW;
            int ppx = __float2int_rd(fmaf(ndcX, 0.5f, 0.5f) * hybridCst.screenWidth);
            int ppy = __float2int_rd(fmaf(ndcY, 0.5f, 0.5f) * hybridCst.screenHeight);
            if (ppx >= 0 && ppx < hybridCst.screenWidth &&
                ppy >= 0 && ppy < hybridCst.screenHeight) {
                int pixelIdx = ppy * hybridCst.screenWidth + ppx;
                unsigned int depthU = __float_as_uint(ndcZ);
                unsigned int old = atomicMin(&depthBuffer[pixelIdx], depthU);
                if (__uint_as_float(old) > ndcZ) {
                    glm::vec3 color = glm::vec3(bondsColor[0], bondsColor[1], bondsColor[2]) * 0.7f * diffuse;
                    uchar4 ucharColor = make_uchar4(
                        (unsigned char)(color.x * 255.0f),
                        (unsigned char)(color.y * 255.0f),
                        (unsigned char)(color.z * 255.0f), 255);
                    surf2Dwrite(ucharColor, outputImage, ppx * sizeof(uchar4), ppy);
                }
            }
        }
        return;
    }

    // ---- Radius clamp for sub-pixel-thick cylinders -------------------------
    // dV1 is the FARTHER endpoint distance (post-swap). Pixel world-size at
    // that depth bounds how thin the projected cylinder can get. The clamp
    // guarantees the closest pixel center is always within `effRadius` of the
    // cylinder axis, removing moire holes along the tube.
    const float effRadius = fmaxf(radius, 0.7071f * dV1 * hybridCst.pxScale);

    for (int py = fminf(hybridCst.screenHeight-1, maxY); py >= fmaxf(0, minY); --py)
    {

        
        float fpy = float(py) + 0.5f; // Centro del pixel en Y
        
        // Encontrar startX y endX para esta fila específica
        // Inicializamos invertidos para acumular min/max
        float rowMinX = float(hybridCst.screenWidth);
        float rowMaxX = -1.0f;
        
        // Recorremos las 4 aristas del polígono proyectado (unroll manual o loop corto)
        #pragma unroll 4
        for (int i = 0; i < 4; ++i) 
        {
            glm::vec2 v1 = projectedPoints[i];
            glm::vec2 v2 = projectedPoints[(i + 1) % 4];

            // Comprobar si la linea cruza esta fila Y
            // Nota: Usamos (>= y <) para evitar doble conteo en vértices exactos
            bool crossing = (v1.y <= fpy && v2.y > fpy) || (v2.y <= fpy && v1.y > fpy);
            
            if (crossing) 
            {
                // Calcular intersección X sin llamadas a funciones externas
                float t = (fpy - v1.y) / (v2.y - v1.y);
                float intersectX = v1.x + t * (v2.x - v1.x);
                
                rowMinX = min(rowMinX, intersectX);
                rowMaxX = max(rowMaxX, intersectX);
            }
        }

        // Convertir span a enteros y clampear
        int startX = max(0, int(floor(rowMinX)));
        int endX   = min(hybridCst.screenWidth - 1, int(ceil(rowMaxX)));

        if (startX <= endX)
        {

            glm::vec3 rayColStart = hybridCst.rayStart + (float)py * hybridCst.dy;
            for (int px = max(0, startX); px <= min(endX, hybridCst.screenWidth -1); ++px)
            {
                glm::vec3 ray = rayColStart + (float)px * hybridCst.dx;
                glm::vec3 rd = fastNormalize(ray.x * hybridCst.right + ray.y * hybridCst.up - ray.z * hybridCst.front);
                glm::vec4 tnor = iCylinder(glm::vec3(hybridCst.cameraPos), rd, pa, pb, effRadius);
                int index = px + hybridCst.screenWidth*py;

                if (tnor.x > 0.0f) {
                    float t = tnor.x;
                    glm::vec3 hit = ray * t;
                    float depth = __fdividef(fmaf(hit.z, hybridCst.proj[2][2], hybridCst.proj[3][2]), -hit.z);

                    unsigned int depthU = __float_as_uint(depth);
                    
                    // atomicMin over uint bits (we store depth as float bits in uint)
                    unsigned int old = atomicMin(&depthBuffer[index], depthU);

                    // if we won (old > depth)
                    if (__uint_as_float(old) > depth) {
                        // own the pixel
                        glm::vec3 normal = fastNormalize(glm::vec3(tnor.y, tnor.z, tnor.w));
                        float lambert = fmaxf(0.0f, glm::dot(normal, -rd));
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
    const glm::vec4* d_spheres,
    const unsigned int* d_smallCylinderIndices,
    unsigned int smallCylinderCount,
    unsigned int* d_depthBuffer,
    cudaSurfaceObject_t outputImage,
    cudaStream_t stream)
{
    if (smallCylinderCount == 0) return;
    dim3 block(256);
    dim3 grid((smallCylinderCount + block.x - 1) / block.x);
    smallCylinderRasterKernel<<<grid, block, 0, stream>>>(d_cylinders, d_spheres, d_smallCylinderIndices, smallCylinderCount, d_depthBuffer, outputImage);
}
