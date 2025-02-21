#include <glad/glad.h>
#include <array>
#include "vis/canvas.h"

Canvas::Canvas(GLenum target, GLenum internalFormat, GLsizei width, 
    GLsizei height, GLenum format, GLenum type):
    m_canvas(target, internalFormat, width, height, format, type, nullptr), m_fbo(),
    m_width(width), m_height(height)
{}

Canvas::Canvas(GLenum target, GLenum internalFormat, GLsizei width, 
    GLsizei height):
    m_canvas(target, internalFormat, width, height), m_fbo(),
    m_width(width), m_height(height)
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

void Canvas::setTextureImage(GLenum target, GLenum internalFormat, GLsizei width, 
    GLsizei height)
{
    m_canvas = Texture(target, internalFormat, width, height);
}

void Canvas::setFBO(GLenum attachment)
{
    m_fbo.attachTexture(attachment, m_canvas, GL_FRAMEBUFFER);
}

void Canvas::bindTexture(GLuint unit) const
{
    m_canvas.bind(unit);
}

void Canvas::bindTextureImage(GLuint unit, GLenum access, GLenum format) const
{
    m_canvas.bindImage(unit, access, format);
}

void Canvas::bindFBO() const
{
    m_fbo.bind(GL_FRAMEBUFFER);
}

Texture& Canvas::getTexture() { return m_canvas; }
Framebuffer& Canvas::getFramebuffer() { return m_fbo; }
GLsizei Canvas::getWidth() const { return m_width; }
GLsizei Canvas::getHeight() const { return m_height; }