#define CUDA_VERSION 13000

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

struct Constants {
    int screenResolutionX;
    int screenResolutionY;
};

static __constant__ Constants cst;

__global__ void downsampleDepthMaxKernel(
    const unsigned int* __restrict__ depthBuffer,
    cudaSurfaceObject_t downsampleSurface)
{

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

            if (x < cst.screenResolutionX && y < cst.screenResolutionY)
            {
                float val = __uint_as_float(__ldg(&depthBuffer[y * cst.screenResolutionX + x]));
                localMax = fmaxf(localMax, val);
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
    int c_screenResolutionX, 
    int c_screenResolutionY, 
    int downsampleWorkGroupSizeX, 
    int downsampleWorkGroupSizeY,
    cudaStream_t& stream
)
{

    Constants h_cst = { c_screenResolutionX, c_screenResolutionY };
    cudaMemcpyToSymbolAsync(cst, &h_cst, sizeof(Constants), 0, cudaMemcpyHostToDevice, stream);

    nvtxRangePushA("Downsampling Kernel");
    dim3 block(4, 4);
    dim3 grid((c_screenResolutionX + downsampleWorkGroupSizeX -1)/downsampleWorkGroupSizeX, (c_screenResolutionY + downsampleWorkGroupSizeY - 1)/downsampleWorkGroupSizeY);
    downsampleDepthMaxKernel<<<grid, block, 0, stream>>>(depthBuffer, downsampleSurface);
    // cudaDeviceSynchronize();
    nvtxRangePop();
}
