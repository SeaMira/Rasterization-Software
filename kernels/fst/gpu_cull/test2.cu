#include <cuda_runtime.h>

__global__ void downsampleDepthMaxKernel(
    const float* depthBuffer,
    int width,
    int height,
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
            if (x < width && y < height)
            {
                float val = depthBuffer[y * width + x];
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


extern "C" void downsamplingDepthTexture(const float* depthBuffer, cudaArray* downsampleArray, int width, int height)
{
    // Crear descriptor del recurso (para el surface object)
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = downsampleArray;

    // Crear surface object
    cudaSurfaceObject_t surfaceObj = 0;
    cudaCreateSurfaceObject(&surfaceObj, &resDesc);

    dim3 block(4,4);
    dim3 grid(width/4, height/4);
    downsampleDepthMaxKernel<<<grid, block>>>(depthBuffer, width, height, surfaceObj);
    cudaDeviceSynchronize();

    cudaDestroySurfaceObject(surfaceObj);
}
