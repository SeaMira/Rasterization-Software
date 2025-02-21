#ifndef _FRAME_BUFFER_H
#define _FRAME_BUFFER_H

#include <glad/glad.h>
#include <iostream>
#include <stdexcept>
#include "vis/gl/texture.h"

class Framebuffer {
public:
    Framebuffer();
    ~Framebuffer();

    void attachTexture(GLenum attachment, const Texture& texture, 
        GLenum framebuffer = GL_FRAMEBUFFER);
    void attachTextureImage(GLenum attachment, const Texture& texture, 
        GLenum framebuffer);
    void bind(GLenum target = GL_FRAMEBUFFER) const;
    void unbind() const;
    bool isComplete() const;

    GLuint getId() const { return m_id; }

private:
    GLuint m_id;
};

#endif