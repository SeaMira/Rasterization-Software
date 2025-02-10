#ifndef _FRAME_BUFFER_H
#define _FRAME_BUFFER_H

#include <glad/glad.h>

#include "vis/gl/texture.h"

class Framebuffer {
public:
    Framebuffer();
    ~Framebuffer();

    void attachTexture(GLenum attachment, const Texture& texture, 
        GLenum framebuffer = GL_FRAMEBUFFER);
    void bind(GLenum target = GL_FRAMEBUFFER) const;
    void unbind() const;
    bool isComplete() const;

private:
    GLuint m_id;
};

#endif