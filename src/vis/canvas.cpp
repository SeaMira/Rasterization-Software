#include <glad/glad.h>
#include <array>
#include <vector>
#include "vis/canvas.h"

Canvas::Canvas(GLenum target, GLenum internalFormat, GLsizei width, 
    GLsizei height, GLenum format, GLenum type):
    m_canvas(target, internalFormat, width, height, format, type, 0, nullptr), m_fbo(),
    m_width(width), m_height(height)
{}

Canvas::Canvas(GLenum target, GLenum internalFormat, GLsizei width, 
    GLsizei height):
    m_canvas(target, internalFormat, width, height, 0), m_fbo(),
    m_width(width), m_height(height)
{}

Canvas::Canvas( Canvas && other ) noexcept
{
    std::swap( m_fbo, other.m_fbo );
    std::swap( m_canvas, other.m_canvas );
    std::swap( m_width, other.m_width );
    std::swap( m_height, other.m_height );
    
}

Canvas & Canvas::operator=( Canvas && other ) noexcept
{
    std::swap( m_fbo, other.m_fbo );
    std::swap( m_canvas, other.m_canvas );
    std::swap( m_width, other.m_width );
    std::swap( m_height, other.m_height );

    return *this;
}

void Canvas::setTexture(GLenum target, GLenum internalFormat, GLsizei width, 
    GLsizei height, GLenum format, GLenum type)
{
    m_canvas = Texture(target, internalFormat, width, height, format, type, 0, nullptr);
}

void Canvas::setTextureImage(GLenum target, GLenum internalFormat, GLsizei width, 
    GLsizei height)
{
    m_canvas = Texture(target, internalFormat, width, height, 0);
}

void Canvas::setFBO(GLenum attachment)
{
    m_fbo.attachTexture(attachment, m_canvas, GL_FRAMEBUFFER);
}

void Canvas::bindTexture() const
{
    m_canvas.bind();
}

void Canvas::bindTextureImage(GLenum access, GLenum format) const
{
    m_canvas.bindImage(access, format);
}

void Canvas::bindFBO() const
{
    m_fbo.bind(GL_FRAMEBUFFER);
}

Texture& Canvas::getTexture() { return m_canvas; }
Framebuffer& Canvas::getFramebuffer() { return m_fbo; }
GLsizei Canvas::getWidth() const { return m_width; }
GLsizei Canvas::getHeight() const { return m_height; }

void flipSurface(SDL_Surface* surface) {
    int pitch = surface->pitch; // Número de bytes por fila
    std::vector<uint8_t> temp(pitch);
    uint8_t* pixels = static_cast<uint8_t*>(surface->pixels);

    for (int y = 0; y < surface->h / 2; y++) {
        uint8_t* row1 = pixels + y * pitch;
        uint8_t* row2 = pixels + (surface->h - y - 1) * pitch;
        std::memcpy(temp.data(), row1, pitch);
        std::memcpy(row1, row2, pitch);
        std::memcpy(row2, temp.data(), pitch);
    }
}

void Canvas::takeScreenshot(std::string screenshot_file) const
{
    bindFBO();
    // std::vector<unsigned char> pixels(m_width * m_height * 4);  
    SDL_Surface* surface = SDL_CreateSurface(m_width, m_height, SDL_PIXELFORMAT_RGBA32);
    if (!surface) {
        std::cerr << "Error: No surface created.\n";
        return;
    }    
    glReadPixels(0, 0, m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, surface->pixels);
    flipSurface(surface);
    bool sshot_saved = SDL_SaveBMP(surface, screenshot_file.c_str());
    if (sshot_saved) std::cout << "Screenshot " << screenshot_file << " saved." << std::endl;
    else std::cout << "Screenshot couldnt be saved." << std::endl;
    SDL_DestroySurface(surface);
    m_fbo.unbind();
}