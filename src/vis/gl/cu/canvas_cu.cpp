#include <glad/glad.h>
#include <array>
#include <vector>
#include "vis/gl/cu/canvas_cu.h"

CanvasCUDA::CanvasCUDA(GLenum target, GLenum internalFormat, GLsizei width, 
    GLsizei height, GLenum format, GLenum type, unsigned int flags):
    m_canvas(target, internalFormat, width, height, format, type, 0, nullptr), m_fbo(),
    m_width(width), m_height(height)
{
    setCUDATextureWrapper(target, flags);
}

CanvasCUDA::CanvasCUDA(GLenum target, GLenum internalFormat, GLsizei width, 
    GLsizei height, unsigned int flags):
    m_canvas(target, internalFormat, width, height, 0), m_fbo(),
    m_width(width), m_height(height)
{
    setCUDATextureWrapper(target, flags);
}

CanvasCUDA::CanvasCUDA( CanvasCUDA && other ) noexcept
{
    std::swap( m_fbo, other.m_fbo );
    std::swap( m_canvas, other.m_canvas );
    std::swap( m_canvasTexWrapper, other.m_canvasTexWrapper );
    std::swap( m_width, other.m_width );
    std::swap( m_height, other.m_height );
    
}

CanvasCUDA & CanvasCUDA::operator=( CanvasCUDA && other ) noexcept
{
    std::swap( m_fbo, other.m_fbo );
    std::swap( m_canvas, other.m_canvas );
    std::swap( m_canvasTexWrapper, other.m_canvasTexWrapper );
    std::swap( m_width, other.m_width );
    std::swap( m_height, other.m_height );

    return *this;
}

void CanvasCUDA::setTexture(GLenum target, GLenum internalFormat, GLsizei width, 
    GLsizei height, GLenum format, GLenum type)
{
    m_canvas = Texture(target, internalFormat, width, height, format, type, 0, nullptr);
}

void CanvasCUDA::setTextureImage(GLenum target, GLenum internalFormat, GLsizei width, 
    GLsizei height)
{
    m_canvas = Texture(target, internalFormat, width, height, 0);
}

void CanvasCUDA::setCUDATextureWrapper(GLenum target, unsigned int flags)
{
    m_canvasTexWrapper.setup(m_canvas, target, flags);
}


void CanvasCUDA::setFBO(GLenum attachment)
{
    m_fbo.attachTexture(attachment, m_canvas, GL_FRAMEBUFFER);
}

void CanvasCUDA::bindTexture() const
{
    m_canvas.bind();
}

void CanvasCUDA::bindTextureImage(GLenum access, GLenum format) const
{
    m_canvas.bindImage(access, format);
}

void CanvasCUDA::bindFBO() const
{
    m_fbo.bind(GL_FRAMEBUFFER);
}

Texture& CanvasCUDA::getTexture() { return m_canvas; }
TextureCUDAWrapper& CanvasCUDA::getCUDATextureWrapper() { return m_canvasTexWrapper; }
DepthDataCUDA& CanvasCUDA::getDepthDataCUDA() { return m_depthData; }
Framebuffer& CanvasCUDA::getFramebuffer() { return m_fbo; }
GLsizei CanvasCUDA::getWidth() const { return m_width; }
GLsizei CanvasCUDA::getHeight() const { return m_height; }

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

void CanvasCUDA::takeScreenshot(std::string screenshot_file) const
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


void CanvasCUDA::setupDepthDataCUDA(int width, int height)
{
    m_depthData.setup(width, height);
    m_isDepthDataSetup = true;
    std::cout << "Depth data setup." << std::endl;
}


void CanvasCUDA::setupDepthDownsampleCUDA(int desiredLevel, int width, int height)
{
    m_depthDownsample.setup(desiredLevel, width, height);
    m_isDepthDownsampleSetup = true;
    std::cout << "Depth downsample setup." << std::endl;
}


// void CanvasCUDA::cleanCanvasBuffers(int workGroupSizeXPerPixel, int workGroupSizeYPerPixel, float far)
// {
//     if (m_isCleaningProgramSetup && m_isDepthDataSetup) 
//     {
//         m_cleaningProgram->use();
//         m_depthData.bindTextureImage(GL_READ_WRITE, GL_R32F);
//         if (m_isDepthDownsampleSetup)
//         {
//             m_depthDownsample.unbindImage(GL_READ_WRITE, GL_R32F);
//             m_depthDownsample.bind();
//         }
//         m_cleaningProgram->setFloat("far", far);
//         m_cleaningProgram->setVec2I("screenResolution", glm::ivec2(m_width, m_height));
//         glDispatchCompute((m_width + workGroupSizeXPerPixel - 1) / workGroupSizeXPerPixel, 
//                 (m_height + workGroupSizeYPerPixel - 1) / workGroupSizeYPerPixel, 
//                 1);
//         glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
//     }
// }

// void CanvasCUDA::downsampleCanvasDepth(int downsampleWorkGroupSizeX, int downsampleWorkGroupSizeY)
// {
//     if (m_isDepthDownsampleSetup && m_isDepthDataSetup && m_isDownsampleProgramSetup) 
//     {
//         m_downsampleProgram->use();
//         m_depthDownsample.bindImage(GL_READ_WRITE, GL_R32F);
//         m_depthData.unbindTextureImage(GL_READ_WRITE, GL_R32F);
//         m_depthData.bindTexture();
//         glm::vec2 utexelDimensions = glm::vec2( 1.0f / (float)m_width, 1.0f / (float)m_height );            
//         m_downsampleProgram->setVec2("utexelDimensions", utexelDimensions);
//         glDispatchCompute((m_width + downsampleWorkGroupSizeX - 1) / downsampleWorkGroupSizeX, 
//             (m_height + downsampleWorkGroupSizeY - 1) / downsampleWorkGroupSizeY, 
//             1);
//         glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
//     }
// }