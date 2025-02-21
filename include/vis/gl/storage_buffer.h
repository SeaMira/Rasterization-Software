#ifndef _STORAGE_BUFFER_H
#define _STORAGE_BUFFER_H

#include <glad/glad.h>
#include "vis/gl/texture.h"

class StorageBuffer
{
public:
    StorageBuffer(GLenum target);
    ~StorageBuffer();
    void generateBufferData(int size, GLuint index,
        const void * data = nullptr, GLenum usage = GL_DYNAMIC_DRAW) const;
    void bind() const;
    void unbind() const;
    bool isComplete() const;

    GLuint getId() const { return m_id; }

private:
    GLuint m_id;
    GLenum m_target;
};

#endif