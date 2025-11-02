#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

__global__ void fillSurfaceKernel(cudaSurfaceObject_t surface,
                                  int width, int height, float time)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    uchar4 color;    
    color = make_uchar4(
    (unsigned char)(127 + 127 * sinf(time + x * 0.01f)),
    (unsigned char)(127 + 127 * sinf(time + y * 0.02f)),
    0, 255); // rojo

    // Escritura directa a la textura 2D
    surf2Dwrite(color, surface, x * sizeof(uchar4), y);
}

extern "C" void writeToTextureCUDA(cudaGraphicsResource* resource, int width, int height, float time) {
    cudaArray* textureArray;
    cudaGraphicsMapResources(1, &resource, 0);
    cudaGraphicsSubResourceGetMappedArray(&textureArray, resource, 0, 0);

    // Crear descriptor del recurso (para el surface object)
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = textureArray;

    // Crear surface object
    cudaSurfaceObject_t surfaceObj = 0;
    cudaCreateSurfaceObject(&surfaceObj, &resDesc);

    dim3 block(16,16);
    dim3 grid((width+15)/16, (height+15)/16);
    fillSurfaceKernel<<<grid, block>>>(surfaceObj, width, height, time);
    cudaDeviceSynchronize();

    cudaDestroySurfaceObject(surfaceObj);
    cudaGraphicsUnmapResources(1, &resource);
}
