#include "vis/gl/depth_data.h"

DepthData::DepthData(int width, int height, int textureBindingPoint, int bufferBindingPoint) :
    m_screenHeight(height),
    m_screenWidth(width),
    m_textureBindingPoint(textureBindingPoint),
    m_bufferBindingPoint(bufferBindingPoint),
    m_texture(GL_TEXTURE_2D, GL_R32F, width, height, textureBindingPoint),
    m_storageBuffer(GL_SHADER_STORAGE_BUFFER, width * height * sizeof(unsigned int), bufferBindingPoint, nullptr, GL_STATIC_DRAW)
{
}

void DepthData::setup(int width, int height, int textureBindingPoint, int bufferBindingPoint)
{
    m_screenHeight = height;
    m_screenWidth = width;
    m_textureBindingPoint = textureBindingPoint;
    m_bufferBindingPoint = bufferBindingPoint;
    m_texture.setup(GL_TEXTURE_2D, GL_R32F, width, height, textureBindingPoint);
    m_storageBuffer.setup(GL_SHADER_STORAGE_BUFFER, width * height * sizeof(unsigned int), bufferBindingPoint, nullptr, GL_STATIC_DRAW);
}


void DepthData::bindTexture()
{
    m_texture.bind();
}

void DepthData::unbindTexture()
{
    m_texture.unbind();
}

void DepthData::bindTextureImage(GLenum access, GLenum format)
{
    m_texture.bindImage(access, format);
}

void DepthData::unbindTextureImage(GLenum access, GLenum format)
{
    m_texture.unbindImage(access, format);
}

void DepthData::bindStorageBuffer()
{
    m_storageBuffer.bind();
}

void DepthData::unbindStorageBuffer()
{
    m_storageBuffer.unbind();
}