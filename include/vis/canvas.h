#ifndef _CANVAS_H_
#define _CANVAS_H_

#include "vis/gl/frame_buffer.h"
#include "vis/gl/texture.h"

class Canvas
{
public:
    Canvas() = default;
    Canvas(GLenum target, GLenum internalFormat, GLsizei width, 
        GLsizei height, GLenum format, GLenum type);

    Canvas( const Canvas& ) = delete;
    Canvas& operator=( const Canvas& ) = delete;

    Canvas( Canvas&& ) noexcept;
    Canvas& operator=( Canvas&& ) noexcept;

    ~Canvas() = default;

    Texture& getTexture();
    Framebuffer& getFramebuffer();
    GLsizei getWidth() const;
    GLsizei getHeight() const;

    void setTexture(GLenum target, GLenum internalFormat, GLsizei width, 
        GLsizei height, GLenum format, GLenum type);
    void setFBO(GLenum attachment);
    void bindFBO() const;

private:
    Texture m_canvas;
    Framebuffer m_fbo;
    GLsizei m_width;
    GLsizei m_height;
};

#endif