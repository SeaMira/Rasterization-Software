#include "vis/gl/depth_downsample.h"

DepthDownsample::DepthDownsample(int desiredLevel, int width, int height, int bindingPoint) :
m_desiredLevel(desiredLevel), m_width(width), m_height(height), m_bindingPoint(bindingPoint),
m_downSampledTexture(GL_TEXTURE_2D, GL_R32F, width/(1 << desiredLevel), height/(1 << desiredLevel), bindingPoint)
{
}

void DepthDownsample::setup(int desiredLevel, int width, int height, int bindingPoint)
{
    m_desiredLevel = desiredLevel;
    m_width = width;
    m_height = height;
    m_bindingPoint = bindingPoint;
    m_downSampledTexture.setup(GL_TEXTURE_2D, GL_R32F, width/(1 << desiredLevel), height/(1 << desiredLevel), bindingPoint);
}

void DepthDownsample::bindImage(GLenum access, GLenum format) const
{
    m_downSampledTexture.bindImage(access, format);
}

void DepthDownsample::unbindImage(GLenum access, GLenum format) const
{
    m_downSampledTexture.unbindImage(access, format);
}

void DepthDownsample::bind() const
{
    m_downSampledTexture.bind();
}   

void DepthDownsample::unbind() const
{
    m_downSampledTexture.unbind();
}   

