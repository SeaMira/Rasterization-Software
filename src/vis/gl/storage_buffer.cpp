#include "vis/gl/storage_buffer.h"

StorageBuffer::StorageBuffer(GLenum target)
{
    m_target = target;
    glGenBuffers(1, &m_id);
}

StorageBuffer::~StorageBuffer()
{
    glDeleteBuffers(1, &m_id);
}

void StorageBuffer::generateBufferData(int size, GLuint index,
    const void * data, GLenum usage) const
{
    glBindBuffer(m_target, m_id);
    glBufferData(m_target, size, data, usage);
    glBindBufferBase(m_target, index, m_id);
}

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
