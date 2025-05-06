#include "vis/gl/storage_buffer.h"

StorageBuffer::StorageBuffer(GLenum target)
{
    m_target = target;
    glGenBuffers(1, &m_id);
}

StorageBuffer::StorageBuffer(GLenum target, int size, GLuint index,
    const void * data, GLenum usage)
{
    m_target = target;
    glGenBuffers(1, &m_id);
    generateBufferData(size, index, data, usage);
    unbind();
}

StorageBuffer::~StorageBuffer()
{
    glDeleteBuffers(1, &m_id);
}

void StorageBuffer::setup(GLenum target, int size, GLuint index,
    const void * data, GLenum usage)
{
    m_target = target;
    glGenBuffers(1, &m_id);
    generateBufferData(size, index, data, usage);
    unbind();
}

void StorageBuffer::generateBufferData(int size, GLuint index,
    const void * data, GLenum usage) const
{
    glBindBuffer(m_target, m_id);
    glBufferData(m_target, size, data, usage);
    glBindBufferBase(m_target, index, m_id);
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cerr << "OpenGL Error: " << error << std::endl;
    }
}

// void StorageBuffer::generateBufferStorage(int size, GLuint index,
//     const void * data = nullptr, GLbitfield flags) const
// {
//     glBindBuffer(m_target, m_id);
//     glBufferStorage(m_target, size, data, flags);
//     glBindBufferBase(m_target, index, m_id);
//     GLenum error = glGetError();
//     if (error != GL_NO_ERROR) {
//         std::cerr << "OpenGL Error: " << error << std::endl;
//     }
// }

void StorageBuffer::bind() const
{
    glBindBuffer(m_target, m_id);
}

void StorageBuffer::unbind() const
{
    glBindBuffer(m_target, 0);
}

bool StorageBuffer::isComplete() const
{
    return glIsBuffer(m_id);
}
