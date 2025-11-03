#include "vis/gl/cu/storage_buffer_cu.h"

StorageBufferCUDAWrapper::StorageBufferCUDAWrapper() :
    m_buffer(nullptr),
    m_cudaResource(nullptr)
{}

StorageBufferCUDAWrapper::StorageBufferCUDAWrapper(StorageBuffer& buffer, unsigned int flags) :
    m_buffer(&buffer),
    m_cudaResource(nullptr)
{
    std::cout << "setting buffer wrapper number: " << buffer.getId() << std::endl;
    cudaRegisterBuffer(flags);
}

StorageBufferCUDAWrapper::~StorageBufferCUDAWrapper()
{
    if (m_cudaResource)
    {
        CUDA_CHECK(cudaGraphicsUnregisterResource(m_cudaResource));
        m_cudaResource = nullptr;
    }
}

void StorageBufferCUDAWrapper::setup(StorageBuffer& buffer, unsigned int flags)
{
    m_buffer = &buffer;
    cudaRegisterBuffer(flags);
}

void StorageBufferCUDAWrapper::cudaRegisterBuffer(unsigned int flags)
{
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&m_cudaResource, m_buffer->getId(), flags));
}

void StorageBufferCUDAWrapper::cudaMapResources()
{
    CUDA_CHECK(cudaGraphicsMapResources(1, &m_cudaResource, 0));
}

void StorageBufferCUDAWrapper::cudaUnmapResources()
{
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_cudaResource, 0));
}



// For template linkage
// template float* StorageBufferCUDAWrapper::getDevicePointer<float>(size_t*);
// template int* StorageBufferCUDAWrapper::getDevicePointer<int>(size_t*);
