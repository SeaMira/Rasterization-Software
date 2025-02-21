#include "vis/gl/frame_buffer.h"

Framebuffer::Framebuffer() {
    glGenFramebuffers(1, &m_id);
}

Framebuffer::~Framebuffer() {
    glDeleteFramebuffers(1, &m_id);
}

void Framebuffer::attachTexture(GLenum attachment, const Texture& texture, 
    GLenum framebuffer) 
{
    glBindFramebuffer(framebuffer, m_id);
    glFramebufferTexture2D(framebuffer, attachment, GL_TEXTURE_2D, texture.getId(), 0);
    GLenum status = glCheckFramebufferStatus(framebuffer);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Framebuffer error: " << status << std::endl;
    }
    glBindFramebuffer(framebuffer, 0);
}

void Framebuffer::bind(GLenum target) const 
{
    glBindFramebuffer(target, m_id);
}

void Framebuffer::unbind() const 
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

bool Framebuffer::isComplete() const 
{
    glBindFramebuffer(GL_FRAMEBUFFER, m_id);
    bool complete = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return complete;
}
