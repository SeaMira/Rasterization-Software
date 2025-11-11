#define CUDA_VERSION 13000

#include <cuda_runtime.h>

__global__ void downsampleDepthMaxKernel(
    unsigned int* depthBuffer,
    int screenResolutionX,
    int screenResolutionY,
    cudaSurfaceObject_t downsampleSurface)
{
    // __shared__ unsigned int sharedMax;
    // if (threadIdx.x == 0 && threadIdx.y == 0)
    //     sharedMax = __float_as_uint(0.0f);
    // __syncthreads();

    // int baseX = blockIdx.x * 4;
    // int baseY = blockIdx.y * 4;

    // float localMax = 0.0f;
    // for (int dy = 0; dy < 4; ++dy)
    // {
    //     for (int dx = 0; dx < 4; ++dx)
    //     {
    //         int x = baseX + dx;
    //         int y = baseY + dy;
    //         if (x < screenResolutionX && y < screenResolutionY)
    //         {
    //             float val = __uint_as_float(depthBuffer[y * screenResolutionX + x]);
    //             if (val > localMax) localMax = val;
    //         }
    //     }
    // }

    // atomicMax(&sharedMax, __float_as_uint(localMax));
    // __syncthreads();

    // if (threadIdx.x == 0 && threadIdx.y == 0)
    // {
    //     float finalMax = __uint_as_float(sharedMax);
    //     surf2Dwrite(finalMax, downsampleSurface, blockIdx.x * sizeof(float), blockIdx.y);
    // }

    // Workgroup de 4x4 threads → procesa 16x16 píxeles
    const int localBlockSize = 4;  // número de threads por dimensión
    const int pixelsPerThread = 4; // cada hilo procesa un bloque 4x4 = total 16x16

    __shared__ unsigned int sharedMax;
    if (threadIdx.x == 0 && threadIdx.y == 0)
        sharedMax = __float_as_uint(0.0f);
    __syncthreads();

    int groupX = blockIdx.x;
    int groupY = blockIdx.y;

    int baseX = groupX * (localBlockSize * pixelsPerThread);
    int baseY = groupY * (localBlockSize * pixelsPerThread);

    float localMax = 0.0f;

    // Cada hilo procesa 4x4 píxeles de su bloque de 16x16
    for (int dy = 0; dy < pixelsPerThread; ++dy)
    {
        for (int dx = 0; dx < pixelsPerThread; ++dx)
        {
            int x = baseX + threadIdx.x * pixelsPerThread + dx;
            int y = baseY + threadIdx.y * pixelsPerThread + dy;

            if (x < screenResolutionX && y < screenResolutionY)
            {
                float val = __uint_as_float(depthBuffer[y * screenResolutionX + x]);
                if (val > localMax)
                    localMax = val;
            }
        }
    }

    // Reducir el máximo dentro del bloque
    atomicMax(&sharedMax, __float_as_uint(localMax));
    __syncthreads();

    // El hilo (0,0) escribe el valor final al surface
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        float finalMax = __uint_as_float(sharedMax);
        surf2Dwrite(finalMax, downsampleSurface,
                    groupX * sizeof(float), groupY);
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
    dim3 block(4, 4);
    dim3 grid((screenResolutionX + downsampleWorkGroupSizeX -1)/downsampleWorkGroupSizeX, (screenResolutionY + downsampleWorkGroupSizeY - 1)/downsampleWorkGroupSizeY);
    downsampleDepthMaxKernel<<<grid, block>>>(depthBuffer, screenResolutionX, screenResolutionY, downsampleSurface);
    cudaDeviceSynchronize();
}
