#define CUDA_VERSION 13000

#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

struct FramePixelsCount {
    unsigned int frames;
    unsigned int pixels;
};

__global__ void pixelCountKernel(
        unsigned int* pixelOwnershipBuffer,   // length = screenW*screenH
        FramePixelsCount* visibilityFrameBuffer,
        int screenResolutionX,
        int screenResolutionY
    )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= screenResolutionX || y >= screenResolutionY) return;

    unsigned int index= pixelOwnershipBuffer[x + screenResolutionX * y];
    visibilityFrameBuffer[index].pixels += 1;
}

extern "C" void pixelCount(
    unsigned int* pixelOwnershipBuffer,   // length = screenW*screenH
    unsigned int* d_visibilityFrameBuffer,
    int workGroupSizeX,
    int workGroupSizeY,
    int screenResolutionX,
    int screenResolutionY
    ) {

    dim3 block(workGroupSizeX, workGroupSizeY);
    dim3 grid((screenResolutionX+workGroupSizeX-1)/workGroupSizeX, (screenResolutionY+workGroupSizeY-1)/workGroupSizeY);
    pixelCountKernel<<<grid, block>>>(
        pixelOwnershipBuffer, 
        (FramePixelsCount*)d_visibilityFrameBuffer, 
        screenResolutionX, 
        screenResolutionY
    );
    cudaDeviceSynchronize();
}