#include "vis/gl/depth_pyramid.h"

DepthPyramid::DepthPyramid(int maxLevels, int width, int height) :
    m_maxLevels(maxLevels), m_width(width), m_height(height)
{
    glGenTextures(1, &m_hiZTexture);
    bindTexture();
    glTexStorage2D(GL_TEXTURE_2D, maxLevels, GL_R32F, width, height);
}

void DepthPyramid::bindLevel(int bindingPoint, int level, GLenum access) const
{
    glBindImageTexture(bindingPoint, m_hiZTexture, level, GL_FALSE, 0, access, GL_R32F);
}

void DepthPyramid::bindTexture() const
{
    glBindTexture(GL_TEXTURE_2D, m_hiZTexture);
}

void DepthPyramid::unbindTexture() const
{
    glBindTexture(GL_TEXTURE_2D, 0);
}   

DepthPyramid::~DepthPyramid()
{
    glBindTexture(GL_TEXTURE_2D, 0);
    glDeleteTextures(1, &m_hiZTexture);
}