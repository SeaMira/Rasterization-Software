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
    std::cout << "Registering texture (ID: " << m_texture->getId() << ") for CUDA access." << std::endl;
    CUDA_CHECK(cudaGraphicsGLRegisterImage(&m_cudaResource, m_texture->getId(), m_target,
                                flags));
}

Texture* TextureCUDAWrapper::getTextureRef() const
{
    std::cout << "Getting texture reference (ID: " << m_texture->getId() << ")." << std::endl;
    return m_texture;
}
void TextureCUDAWrapper::setTexture(Texture& texture)
{
    std::cout << "Setting texture (ID: " << texture.getId() << ") for CUDA access." << std::endl;
    m_texture = &texture;
}

GLuint TextureCUDAWrapper::getTextureId() const
{
    std::cout << "Getting texture ID (ID: " << m_texture->getId() << ")." << std::endl;
    return m_texture->getId();
}

void TextureCUDAWrapper::setTarget(GLenum target)
{
    std::cout << "Setting texture target " << target << " (ID: " << m_texture->getId() << ") for CUDA access." << std::endl;
    m_target = target;
}

GLenum TextureCUDAWrapper::getTarget() const
{
    std::cout << "Getting texture target " << m_target << " (ID: " << m_texture->getId() << ")." << std::endl;
    return m_target;
}

void TextureCUDAWrapper::cudaMapResources()
{
    // std::cout << "Mapping resources for texture (ID: " << m_texture->getId() << ")." << std::endl;
    CUDA_CHECK(cudaGraphicsMapResources(1, &m_cudaResource, 0));
    CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&m_textureArray, m_cudaResource, 0, 0));
}

void TextureCUDAWrapper::cudaCreateSurfaceObj()
{
    // std::cout << "Creating surface object for texture (ID: " << m_texture->getId() << ")." << std::endl;
    m_resDesc.resType = cudaResourceTypeArray;
    m_resDesc.res.array.array = m_textureArray;
    CUDA_CHECK(cudaCreateSurfaceObject(&m_surfaceObj, &m_resDesc));
}

void TextureCUDAWrapper::cudaDestroySurfaceObj()
{
    // std::cout << "Destroying surface object for texture (ID: " << m_texture->getId() << ")." << std::endl;
    CUDA_CHECK(cudaDestroySurfaceObject(m_surfaceObj));
    m_surfaceObj = 0;
}

void TextureCUDAWrapper::cudaUnmapResources()
{
    // std::cout << "Unmapping resources for texture (ID: " << m_texture->getId() << ")." << std::endl;
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_cudaResource));
}

void TextureCUDAWrapper::cudaUnregisterTex()
{
    std::cout << "Unregistering texture (ID: " << m_texture->getId() << ") from CUDA access." << std::endl;
    if (m_cudaResource) {
        CUDA_CHECK(cudaGraphicsUnregisterResource(m_cudaResource));
        m_cudaResource = nullptr;
    }
}

cudaSurfaceObject_t TextureCUDAWrapper::getSurfaceObject() const
{
    // std::cout << "Getting surface object for texture (ID: " << m_texture->getId() << ")." << std::endl;
    return m_surfaceObj;
}