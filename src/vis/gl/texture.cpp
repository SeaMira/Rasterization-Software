#include "vis/gl/texture.h"

Texture::Texture(GLenum target, GLenum internalFormat, GLsizei width, GLsizei height, GLenum format, GLenum type, void* data) 
    : m_target(target) {
    glGenTextures(1, &m_id);
    glBindTexture(target, m_id);
    glTexImage2D(target, 0, internalFormat, width, height, 0, format, type, data);
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindImageTexture(0, m_id, 0, GL_FALSE, 0, GL_WRITE_ONLY, internalFormat);
    glBindTexture(target, 0);
}

Texture::~Texture() {
    glDeleteTextures(1, &m_id);
}

void Texture::bind(GLuint unit) const {
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(m_target, m_id);
}

void Texture::unbind() const {
    glBindTexture(m_target, 0);
}

GLuint Texture::getId() const {
    return m_id;
}