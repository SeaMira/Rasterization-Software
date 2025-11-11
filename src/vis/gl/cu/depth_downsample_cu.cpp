#include "vis/gl/cu/depth_downsample_cu.h"



DepthDownsampleCUDA::DepthDownsampleCUDA()
    : m_desiredLevel(0), m_width(0), m_height(0),
      m_downsampleArray(nullptr),
      m_surface(0), m_texture(0)
{
}


DepthDownsampleCUDA::DepthDownsampleCUDA(int desiredLevel, int width, int height)
{
    setup(desiredLevel, width, height);
}


DepthDownsampleCUDA::~DepthDownsampleCUDA()
{
    if (m_texture)  CUDA_CHECK(cudaDestroyTextureObject(m_texture));
    if (m_surface)  CUDA_CHECK(cudaDestroySurfaceObject(m_surface));
    if (m_downsampleArray) CUDA_CHECK(cudaFreeArray(m_downsampleArray));
}


void DepthDownsampleCUDA::setup(int desiredLevel, int width, int height)
{
    m_desiredLevel = desiredLevel;
    m_width = width/(1 << desiredLevel);
    m_height = height/(1 << desiredLevel);

    // Allocate CUDA resources
    memoryAlloc();
    writingSurfaceSetup();
    readingSurfaceSetup();
}


void DepthDownsampleCUDA::setup()
{
    // Default setup values
    setup(m_desiredLevel, m_width, m_height);
}

void DepthDownsampleCUDA::resolutionSetup(int width, int height)
{
    m_width = width/(1 << m_desiredLevel);
    m_height = height/(1 << m_desiredLevel);
}

void DepthDownsampleCUDA::downsamplingLevelSetup(int desiredLevel)
{
    m_desiredLevel = desiredLevel;
}

void DepthDownsampleCUDA::memoryAlloc()
{
    // Allocate CUDA array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    CUDA_CHECK(cudaMallocArray(&m_downsampleArray, &channelDesc, m_width, m_height, cudaArraySurfaceLoadStore));
}

void DepthDownsampleCUDA::writingSurfaceSetup()
{
    // Create surface object
    m_surfResDesc.resType = cudaResourceTypeArray;
    m_surfResDesc.res.array.array = m_downsampleArray;
    CUDA_CHECK(cudaCreateSurfaceObject(&m_surface, &m_surfResDesc));

}

void DepthDownsampleCUDA::readingSurfaceSetup()
{
    // Create texture object
    m_texDesc.addressMode[0] = cudaAddressModeClamp;
    m_texDesc.addressMode[1] = cudaAddressModeClamp;
    m_texDesc.filterMode = cudaFilterModePoint;
    m_texDesc.readMode = cudaReadModeElementType;
    m_texDesc.normalizedCoords = 0;
    CUDA_CHECK(cudaCreateTextureObject(&m_texture, &m_surfResDesc, &m_texDesc, nullptr));
}

bool DepthDownsampleCUDA::isValid() const
{
    return (m_downsampleArray != nullptr) && (m_surface != 0) && (m_texture != 0);
}

cudaSurfaceObject_t DepthDownsampleCUDA::getSurface() const
{
    return m_surface;
}

cudaTextureObject_t DepthDownsampleCUDA::getTexture() const
{
    return m_texture;
}