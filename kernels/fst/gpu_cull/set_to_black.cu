#define CUDA_VERSION 13000

#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <nvtx3/nvToolsExt.h>
#include "utils/scene_descriptor.h"

struct Constants {
    float c_far;
    int screenResolutionX;
    int screenResolutionY;
};

static __constant__ Constants cst;


__global__ void cleaningKernel(
        cudaSurfaceObject_t surface,
        unsigned int* __restrict__ depthBuffer,
        unsigned int* __restrict__ pixelOwnershipBuffer
    )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cst.screenResolutionX || y >= cst.screenResolutionY) return;

    pixelOwnershipBuffer[x + cst.screenResolutionX * y] = 0;
    depthBuffer[x + cst.screenResolutionX * y] = __float_as_uint(cst.c_far);

    uchar4 color;
    color = make_uchar4(backgroundColor.x * 255, backgroundColor.y * 255, backgroundColor.z * 255, 255); 

    // Escritura directa a la textura 2D
    surf2Dwrite(color, surface, x * sizeof(uchar4), y);
}

extern "C" void cleaningScreen(
    cudaSurfaceObject_t texSurfaceObj, 
    unsigned int* depthBuffer,
    unsigned int* pixelOwnershipBuffer,
    int workGroupSizeXPerPixel,
    int workGroupSizeYPerPixel,
    float c_far, 
    int c_screenResolutionX,
    int c_screenResolutionY,
    cudaStream_t& stream
    ) {
    

    Constants h_cst = { c_far, c_screenResolutionX, c_screenResolutionY };
    cudaMemcpyToSymbolAsync(cst, &h_cst, sizeof(Constants), 0, cudaMemcpyHostToDevice, stream);

    nvtxRangePushA("Cleaning Kernel");
    dim3 block(workGroupSizeXPerPixel,workGroupSizeYPerPixel);
    dim3 grid((c_screenResolutionX+workGroupSizeXPerPixel-1)/workGroupSizeXPerPixel, (c_screenResolutionY+workGroupSizeYPerPixel-1)/workGroupSizeYPerPixel);
    cleaningKernel<<<grid, block, 0, stream>>>(texSurfaceObj, depthBuffer, pixelOwnershipBuffer);
    // cudaDeviceSynchronize();
    nvtxRangePop();
}