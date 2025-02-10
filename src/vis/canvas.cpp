#include <glad/glad.h>
#include <array>
#include "vis/canvas.h"

Canvas::Canvas(GLenum target, GLenum internalFormat, GLsizei width, 
    GLsizei height, GLenum format, GLenum type):
    m_canvas(target, internalFormat, width, height, format, type, nullptr)
{}

Canvas::Canvas( Canvas && other ) noexcept
{
    std::swap( m_fbo, other.m_fbo );
    std::swap( m_canvas, other.m_canvas );
    std::swap( m_width, other.m_width );
    std::swap( m_height, other.m_height );
    
}

Canvas & Canvas::operator=( Canvas && other ) noexcept
{
    std::swap( m_fbo, other.m_fbo );
    std::swap( m_canvas, other.m_canvas );
    std::swap( m_width, other.m_width );
    std::swap( m_height, other.m_height );

    return *this;
}

void Canvas::setTexture(GLenum target, GLenum internalFormat, GLsizei width, 
    GLsizei height, GLenum format, GLenum type)
{
    m_canvas = Texture(target, internalFormat, width, height, format, type, nullptr);
}

void Canvas::setFBO(GLenum attachment)
{
    m_fbo.attachTexture(attachment, m_canvas, GL_READ_FRAMEBUFFER);
}

void Canvas::bindFBO() const
{
    m_fbo.bind(GL_READ_FRAMEBUFFER);
}

Texture& Canvas::getTexture() { return m_canvas; }
Framebuffer& Canvas::getFramebuffer() { return m_fbo; }
GLsizei Canvas::getWidth() const { return m_width; }
GLsizei Canvas::getHeight() const { return m_height; }