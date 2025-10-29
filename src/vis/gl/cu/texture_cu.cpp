#include "vis/gl/cu/texture_cu.h"



TextureCUDAWrapper::TextureCUDAWrapper(): 
    m_target(GL_NONE), m_texture(nullptr),
    m_cudaResource(nullptr), m_textureArray(nullptr),
    m_resDesc({}), m_surfaceObj(0)
{
}

TextureCUDAWrapper::TextureCUDAWrapper(Texture& texture, GLenum target, unsigned int flags):
    m_target(target), m_texture(&texture),
    m_cudaResource(nullptr), m_textureArray(nullptr),
    m_resDesc({}), m_surfaceObj(0)
{
    cudaRegisterTex(flags);
}

TextureCUDAWrapper::~TextureCUDAWrapper()
{
    if (m_surfaceObj) 
        cudaDestroySurfaceObj();

    if (m_cudaResource) 
        cudaUnregisterTex();
}


void TextureCUDAWrapper::setup(Texture& texture, GLenum target, unsigned int flags)
{
    m_target = target;
    m_texture = &texture;
    cudaRegisterTex(flags);
}

void TextureCUDAWrapper::cudaRegisterTex(unsigned int flags)
{
    CUDA_CHECK(cudaGraphicsGLRegisterImage(&m_cudaResource, m_texture->getId(), m_target,
                                flags));
}

Texture* TextureCUDAWrapper::getTextureRef() const
{
    return m_texture;
}
void TextureCUDAWrapper::setTexture(Texture& texture)
{
    m_texture = &texture;
}

GLuint TextureCUDAWrapper::getTextureId() const
{
    return m_texture->getId();
}

void TextureCUDAWrapper::setTarget(GLenum target)
{
    m_target = target;
}

GLenum TextureCUDAWrapper::getTarget() const
{
    return m_target;
}

void TextureCUDAWrapper::cudaMapResources()
{
    CUDA_CHECK(cudaGraphicsMapResources(1, &m_cudaResource, 0));
    CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&m_textureArray, m_cudaResource, 0, 0));
}

void TextureCUDAWrapper::cudaCreateSurfaceObj()
{
    m_resDesc.resType = cudaResourceTypeArray;
    m_resDesc.res.array.array = m_textureArray;
    CUDA_CHECK(cudaCreateSurfaceObject(&m_surfaceObj, &m_resDesc));
}

void TextureCUDAWrapper::cudaDestroySurfaceObj()
{
    CUDA_CHECK(cudaDestroySurfaceObject(m_surfaceObj));
    m_surfaceObj = 0;
}

void TextureCUDAWrapper::cudaUnmapResources()
{
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_cudaResource, 0));
}

void TextureCUDAWrapper::cudaUnregisterTex()
{
    if (m_cudaResource) {
        CUDA_CHECK(cudaGraphicsUnregisterResource(m_cudaResource));
        m_cudaResource = nullptr;
    }
}

cudaSurfaceObject_t TextureCUDAWrapper::getSurfaceObject() const
{
    return m_surfaceObj;
}