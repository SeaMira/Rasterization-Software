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

Texture::Texture(GLenum target, GLenum internalFormat, GLsizei width, GLsizei height) 
    : m_target(target) {
    glCreateTextures(target, 1, &m_id);

    if (m_id == 0) {
        std::cerr << "Error: No se pudo generar la textura." << std::endl;
    } else {
        std::cout << "Textura generada correctamente con ID: " << m_id << std::endl;
    }

    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexStorage2D(target, 1, internalFormat, width, height);
    glBindImageTexture(0, m_id, 0, GL_FALSE, 0, GL_READ_WRITE, internalFormat);
    glBindTexture(target, 0);
}

Texture::~Texture() {
    glDeleteTextures(1, &m_id);
}

void Texture::bind(GLuint unit) const {
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(m_target, m_id);
}

void Texture::bindImage(GLuint unit, GLenum access, GLenum format) const {
    glBindImageTexture(unit, m_id, 0, GL_FALSE, 0, access, format);
}

void Texture::unbind() const {
    glBindTexture(m_target, 0);
}

GLuint Texture::getId() const {
    return m_id;
}