#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

__global__ void cleaningKernel(
        cudaSurfaceObject_t surface,
        unsigned int* depthBuffer,
        unsigned int* pixelOwnershipBuffer,
        float far, 
        float screenResolutionX,
        float screenResolutionY
    )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= screenResolutionX || y >= screenResolutionY) return;

    pixelOwnershipBuffer[x + screenResolutionX * y] = 0;
    depthBuffer[x + screenResolutionX * y] = __float_as_uint(far);

    uchar4 color;
    color = make_uchar4(255, 255, 0, 255); // amarillo

    // Escritura directa a la textura 2D
    surf2Dwrite(color, surface, x * sizeof(uchar4), y);
}

extern "C" void cleaningScreen(
    cudaSurfaceObject_t& texSurfaceObj, 
    unsigned int* depthBuffer,
    unsigned int* pixelOwnershipBuffer,
    int workGroupSizeXPerPixel,
    int workGroupSizeYPerPixel,
    float far, 
    float screenResolutionX,
    float screenResolutionY
    ) {

    dim3 block(workGroupSizeXPerPixel,workGroupSizeYPerPixel);
    dim3 grid((width+workGroupSizeXPerPixel-1)/workGroupSizeXPerPixel, (height+workGroupSizeYPerPixel-1)/workGroupSizeYPerPixel);
    fillSurfaceKernel<<<grid, block>>>(texSurfaceObj, depthBuffer, pixelOwnershipBuffer, far, screenResolutionX, screenResolutionY);
    cudaDeviceSynchronize();
}