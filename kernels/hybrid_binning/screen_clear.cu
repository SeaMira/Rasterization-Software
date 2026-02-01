/**
 * @file screen_clear.cu
 * @brief Screen clearing for hybrid binning pipeline.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void clearScreenKernel(
    cudaSurfaceObject_t outputImage,
    unsigned int* __restrict__ depthBuffer,
    unsigned int screenWidth,
    unsigned int screenHeight,
    float farPlane)
{
    unsigned int px = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= screenWidth || py >= screenHeight) return;
    uchar4 clearColor = make_uchar4(0, 0, 0, 255);
    surf2Dwrite(clearColor, outputImage, px * sizeof(uchar4), py);
    depthBuffer[py * screenWidth + px] = __float_as_uint(farPlane);
}

extern "C" void launchScreenClear(
    cudaSurfaceObject_t outputImage,
    unsigned int* d_depthBuffer,
    unsigned int screenWidth,
    unsigned int screenHeight,
    float farPlane,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((screenWidth + block.x - 1) / block.x, (screenHeight + block.y - 1) / block.y);
    clearScreenKernel<<<grid, block, 0, stream>>>(outputImage, d_depthBuffer, screenWidth, screenHeight, farPlane);
}
