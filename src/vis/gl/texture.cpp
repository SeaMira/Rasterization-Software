#include "vis/gl/texture.h"

Texture::Texture(GLenum target, GLenum internalFormat, GLsizei width, GLsizei height, GLenum format, GLenum type, GLuint unit, void* data) 
    : m_target(target), m_unit(unit) {
    glGenTextures(1, &m_id);
    glBindTexture(target, m_id);
    if (m_id == 0) {
        std::cerr << "Error:couldn't generate texture." << std::endl;
    } else {
        std::cout << "Texture correctly generated with ID: " << m_id << std::endl;
    }
    glTexImage2D(target, 0, internalFormat, width, height, 0, format, type, data);
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindImageTexture(unit, m_id, 0, GL_FALSE, 0, GL_WRITE_ONLY, internalFormat);
    glBindTexture(target, 0);
}

Texture::Texture(GLenum target, GLenum internalFormat, GLsizei width, GLsizei height, GLuint unit, GLuint mipLevels) 
: m_target(target), m_unit(unit) {
    glCreateTextures(target, 1, &m_id);

    if (m_id == 0) {
        std::cerr << "Error:couldn't generate texture." << std::endl;
    } else {
        std::cout << "Texture correctly generated with ID: " << m_id << std::endl;
    }
    glBindTexture(target, m_id);
    
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexStorage2D(target, mipLevels, internalFormat, width, height);

    if (internalFormat != GL_DEPTH_COMPONENT &&
        internalFormat != GL_DEPTH_COMPONENT16 &&
        internalFormat != GL_DEPTH_COMPONENT24 && 
        internalFormat != GL_DEPTH_COMPONENT32 && 
        internalFormat != GL_DEPTH_COMPONENT32F && 
        internalFormat != GL_DEPTH_STENCIL
    )
        glBindImageTexture(unit, m_id, 0, GL_FALSE, 0, GL_WRITE_ONLY, internalFormat);
    
    glBindTexture(target, 0);
}

Texture::Texture(GLenum target, GLenum internalFormat, GLsizei width, GLsizei height, 
    GLenum format, GLenum type, GLuint unit, void* data,
    GLint min_filter_param, GLint mag_filter_param, GLint wrap_s_param, GLint wrap_t_param): 
    m_unit(unit)
{
    glCreateTextures(target, 1, &m_id);
    
    if (m_id == 0) {
        std::cerr << "Error:couldn't generate texture." << std::endl;
    } else {
        std::cout << "Texture correctly generated with ID: " << m_id << std::endl;
    }
    glBindTexture(target, m_id);

    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, min_filter_param);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, mag_filter_param);
    glTexParameteri(target, GL_TEXTURE_WRAP_S, wrap_s_param);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, wrap_t_param);
    glTexImage2D(target, 0, internalFormat, width, height, 0, format, type, data);
    glBindImageTexture(unit, m_id, 0, GL_FALSE, 0, GL_WRITE_ONLY, internalFormat);
    glBindTexture(target, 0);
}

Texture::~Texture() {
    if (m_id != 0) {
        glDeleteTextures(1, &m_id);
        GLenum error = glGetError();
        if (error != GL_NO_ERROR) {
            std::cerr << "OpenGL Error: " << error << std::endl;
        } else std::cout << "Texture " << m_id << " deleted succesfully" << std::endl;
    }
}

void Texture::setup(GLenum target, GLenum internalFormat, GLsizei width, GLsizei height, GLenum format, GLenum type, GLuint unit, void* data) 
{
    m_target = target; 
    m_unit = unit;
    glGenTextures(1, &m_id);
    glBindTexture(target, m_id);
    if (m_id == 0) {
        std::cerr << "Error:couldn't generate texture." << std::endl;
    } else {
        std::cout << "Texture correctly generated with ID: " << m_id << std::endl;
    }
    glTexImage2D(target, 0, internalFormat, width, height, 0, format, type, data);
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindImageTexture(unit, m_id, 0, GL_FALSE, 0, GL_WRITE_ONLY, internalFormat);
    glBindTexture(target, 0);
}

void Texture::setup(GLenum target, GLenum internalFormat, GLsizei width, GLsizei height, GLuint unit) 
{
    m_target = target; 
    m_unit = unit;
    glCreateTextures(target, 1, &m_id);

    if (m_id == 0) {
        std::cerr << "Error:couldn't generate texture." << std::endl;
    } else {
        std::cout << "Texture correctly generated with ID: " << m_id << std::endl;
    }
    glBindTexture(target, m_id);
    
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexStorage2D(target, 1, internalFormat, width, height);
    glBindImageTexture(unit, m_id, 0, GL_FALSE, 0, GL_WRITE_ONLY, internalFormat);
    glBindTexture(target, 0);
}

void Texture::setup(GLenum target, GLenum internalFormat, GLsizei width, GLsizei height, 
    GLenum format, GLenum type, GLuint unit, void* data,
    GLint min_filter_param, GLint mag_filter_param, GLint wrap_s_param, GLint wrap_t_param)
{
    m_unit = unit;
    glCreateTextures(target, 1, &m_id);
    
    if (m_id == 0) {
        std::cerr << "Error:couldn't generate texture." << std::endl;
    } else {
        std::cout << "Texture correctly generated with ID: " << m_id << std::endl;
    }
    glBindTexture(target, m_id);

    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, min_filter_param);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, mag_filter_param);
    glTexParameteri(target, GL_TEXTURE_WRAP_S, wrap_s_param);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, wrap_t_param);
    glTexImage2D(target, 0, internalFormat, width, height, 0, format, type, data);
    glBindImageTexture(unit, m_id, 0, GL_FALSE, 0, GL_WRITE_ONLY, internalFormat);
    glBindTexture(target, 0);
}

void Texture::bind() const {
    glActiveTexture(GL_TEXTURE0 + m_unit);
    glBindTexture(m_target, m_id);
}

void Texture::bindImage(GLenum access, GLenum format) const {
    glBindImageTexture(m_unit, m_id, 0, GL_FALSE, 0, access, format);
}

void Texture::unbindImage(GLenum access, GLenum format) const {
    glBindImageTexture(m_unit, 0, 0, GL_FALSE, 0, access, format);
}

void Texture::unbind() const {
    glBindTexture(m_target, 0);
}

GLuint Texture::getId() const {
    return m_id;
}