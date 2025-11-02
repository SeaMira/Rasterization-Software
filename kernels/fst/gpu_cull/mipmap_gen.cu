#include <cuda_runtime.h>

__global__ void downsampleDepthMaxKernel(
    unsigned int* depthBuffer,
    int screenResolutionX,
    int screenResolutionY,
    cudaSurfaceObject_t downsampleSurface)
{
    __shared__ unsigned int sharedMax;
    if (threadIdx.x == 0 && threadIdx.y == 0)
        sharedMax = __float_as_uint(0.0f);
    __syncthreads();

    int baseX = blockIdx.x * 4;
    int baseY = blockIdx.y * 4;

    float localMax = 0.0f;
    for (int dy = 0; dy < 4; ++dy)
    {
        for (int dx = 0; dx < 4; ++dx)
        {
            int x = baseX + dx;
            int y = baseY + dy;
            if (x < screenResolutionX && y < screenResolutionY)
            {
                float val = depthBuffer[y * screenResolutionX + x];
                if (val > localMax) localMax = val;
            }
        }
    }

    atomicMax(&sharedMax, __float_as_uint(localMax));
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        float finalMax = __uint_as_float(sharedMax);
        surf2Dwrite(finalMax, downsampleSurface, blockIdx.x * sizeof(float), blockIdx.y);
    }
}


extern "C" void downsamplingDepthTexture(
    unsigned int* depthBuffer, 
    cudaSurfaceObject_t downsampleSurface, 
    int screenResolutionX, 
    int screenResolutionY, 
    int downsampleWorkGroupSizeX, 
    int downsampleWorkGroupSizeY)
{
    dim3 block(downsampleWorkGroupSizeX, downsampleWorkGroupSizeY);
    dim3 grid((screenResolutionX + downsampleWorkGroupSizeX -1)/downsampleWorkGroupSizeX, (downsampleWorkGroupSizeY + downsampleWorkGroupSizeY - 1)/downsampleWorkGroupSizeY);
    downsampleDepthMaxKernel<<<grid, block>>>(depthBuffer, screenResolutionX, screenResolutionY, downsampleSurface);
    cudaDeviceSynchronize();
}
