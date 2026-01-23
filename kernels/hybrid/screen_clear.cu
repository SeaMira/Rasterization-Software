/**
 * @file screen_clear.cu
 * @brief Screen clearing kernel for the hybrid pipeline
 */

#define GLM_FORCE_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ============================================================================
// Screen Clearing Kernel
// ============================================================================

__global__ void clearScreenKernel(
    cudaSurfaceObject_t outputImage,
    unsigned int* __restrict__ depthBuffer,
    unsigned int screenWidth,
    unsigned int screenHeight,
    float farPlane)
{
    const unsigned int px = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int py = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (px >= screenWidth || py >= screenHeight) return;
    
    // Clear color to black
    uchar4 clearColor = make_uchar4(0, 0, 0, 255);
    surf2Dwrite(clearColor, outputImage, px * sizeof(uchar4), py);
    
    // Clear depth to far plane
    unsigned int pixelIdx = py * screenWidth + px;
    depthBuffer[pixelIdx] = __float_as_uint(farPlane);
}

// ============================================================================
// Host wrapper
// ============================================================================

extern "C" void launchScreenClear(
    cudaSurfaceObject_t outputImage,
    unsigned int* d_depthBuffer,
    unsigned int screenWidth,
    unsigned int screenHeight,
    float farPlane,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((screenWidth + block.x - 1) / block.x,
              (screenHeight + block.y - 1) / block.y);
    
    clearScreenKernel<<<grid, block, 0, stream>>>(
        outputImage, d_depthBuffer, screenWidth, screenHeight, farPlane);
}
