#ifndef _TEXTURE_H_
#define _TEXTURE_H_

#include <glad/glad.h>

class Texture {
public:
    Texture(GLenum target, GLenum internalFormat, GLsizei width, GLsizei height, 
        GLenum format, GLenum type, void* data = nullptr);
    Texture() = default;
    ~Texture();

    void bind(GLuint unit = 0) const;
    void unbind() const;
    GLuint getId() const;

private:
    GLuint m_id;
    GLenum m_target;
    friend class Canvas;
};

#endif