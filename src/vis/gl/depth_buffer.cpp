#include "vis/gl/depth_buffer.h"

DepthBuffer::DepthBuffer(int width, int height, int bindingPoint) :
    m_texture(GL_TEXTURE_2D, GL_R32F, width, height, 
        GL_RED, GL_FLOAT, (GLuint)bindingPoint, nullptr, 
        GL_NEAREST, GL_NEAREST,
        GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE), m_bindingPoint(bindingPoint)
{
    glGenSamplers(1, &m_depthSampler);

    glSamplerParameteri(m_depthSampler, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glSamplerParameteri(m_depthSampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glSamplerParameteri(m_depthSampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glSamplerParameteri(m_depthSampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}


void DepthBuffer::bind()
{
    m_texture.bind();
    glBindSampler(m_bindingPoint, m_depthSampler);
}

void DepthBuffer::bindImage(GLuint unit, GLenum access)
{
    m_texture.bindImage(access, GL_R32F);
}

void DepthBuffer::unbind()
{
    m_texture.unbind();
    glBindSampler(m_bindingPoint, 0);
}

DepthBuffer::~DepthBuffer()
{
    glDeleteSamplers(1, &m_depthSampler);
}