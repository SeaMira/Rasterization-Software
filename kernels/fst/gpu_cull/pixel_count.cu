#define CUDA_VERSION 13000

#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

struct Constants {
    int screenResolutionX;
    int screenResolutionY;
};

__constant__ Constants cst;

struct FramePixelsCount {
    unsigned int frames;
    unsigned int pixels;
};

__global__ void pixelCountKernel(
        unsigned int* __restrict__ pixelOwnershipBuffer,   // length = screenW*screenH
        FramePixelsCount* __restrict__ visibilityFrameBuffer
    )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cst.screenResolutionX || y >= cst.screenResolutionY) return;

    unsigned int index= pixelOwnershipBuffer[x + cst.screenResolutionX * y];
    visibilityFrameBuffer[index].pixels += 1;
}

extern "C" void pixelCount(
    unsigned int* pixelOwnershipBuffer,   // length = screenW*screenH
    unsigned int* d_visibilityFrameBuffer,
    int workGroupSizeX,
    int workGroupSizeY,
    int c_screenResolutionX,
    int c_screenResolutionY,
    cudaStream_t& stream
    ) {
    Constants h_cst = { c_screenResolutionX, c_screenResolutionY };
    cudaMemcpyToSymbol(cst, &h_cst, sizeof(Constants));

    dim3 block(workGroupSizeX, workGroupSizeY);
    dim3 grid((c_screenResolutionX+workGroupSizeX-1)/workGroupSizeX, (c_screenResolutionY+workGroupSizeY-1)/workGroupSizeY);
    pixelCountKernel<<<grid, block, 0, stream>>>(
        pixelOwnershipBuffer, 
        (FramePixelsCount*)d_visibilityFrameBuffer
    );
    // cudaDeviceSynchronize();
}