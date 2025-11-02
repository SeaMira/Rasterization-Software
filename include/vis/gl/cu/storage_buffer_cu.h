#ifndef _STORAGE_BUFFER_CU_H_
#define _STORAGE_BUFFER_CU_H_

#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdexcept>
#include <iostream>

#include "vis/gl/storage_buffer.h"
#include "utils/parallel/cuda_checks.h"

/**
 * @class StorageBufferCUDAWrapper
 * 
 * @brief Wrapper de interoperabilidad GL–CUDA para SSBOs.
 * 
 * Permite usar un StorageBuffer de OpenGL (SSBO) directamente en CUDA como puntero lineal.
 */
class StorageBufferCUDAWrapper
{
public:
    StorageBufferCUDAWrapper();
    StorageBufferCUDAWrapper(StorageBuffer& buffer, unsigned int flags = cudaGraphicsRegisterFlagsNone);
    ~StorageBufferCUDAWrapper();

    void setup(StorageBuffer& buffer, unsigned int flags = cudaGraphicsRegisterFlagsNone);
    void cudaRegisterBuffer(unsigned int flags = cudaGraphicsRegisterFlagsNone);

    void cudaMapResources();
    void cudaUnmapResources();

    template<typename T>
    T* getDevicePointer(size_t* sizeInBytes = nullptr)
    {
        void* devPtr = nullptr;
        size_t bytes = 0;
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(&devPtr, &bytes, m_cudaResource));
        if (sizeInBytes) *sizeInBytes = bytes;
        return reinterpret_cast<T*>(devPtr);
    }

    StorageBuffer* getBufferRef() const { return m_buffer; }

private:
    StorageBuffer* m_buffer = nullptr;
    cudaGraphicsResource* m_cudaResource = nullptr;
};

#endif
