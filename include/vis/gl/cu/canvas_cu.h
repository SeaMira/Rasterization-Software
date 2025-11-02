#ifndef _CANVAS_H_
#define _CANVAS_H_

#include <SDL3/SDL.h>
#include "vis/gl/frame_buffer.h"
#include "vis/gl/texture.h"
#include "vis/gl/cu/texture_cu.h"
#include "vis/gl/cu/depth_data_cu.h"
#include "vis/gl/cu/depth_downsample_cu.h"

/**
 * @class Canvas 
 * 
 * @brief Canvas class used to set the viewport in which to draw directly.
 * 
 * Used as direct canvas to draw and show in the viewport. It's texture and framebuffer
 * are the main ones to be shown.
 */
class CanvasCUDA
{
public:
    /**
     * @brief Class default constructor.
     * 
     * Class default constructor for future setup.
     */
    CanvasCUDA() = default;

    /**
     * @brief Class constructor.
     * 
     * Class constructor, sets the framebuffers and textures representing the viewport as a canvas.
     * 
     * @param target framebuffer target to set.
     * @param internalFormat image format, specifies the number of color components.
     * @param width width of the texture.
     * @param height height of the texture.
     * @param format format of the pixel data.
     * @param type data type of the pixel.
     */
    CanvasCUDA(GLenum target, GLenum internalFormat, GLsizei width, 
        GLsizei height, GLenum format, GLenum type, unsigned int flags=cudaGraphicsRegisterFlagsSurfaceLoadStore);

    /**
     * @brief Class constructor.
     * 
     * Class constructor, sets the framebuffers and textures representing the viewport as a canvas.
     * 
     * @param target framebuffer target to set.
     * @param internalFormat image format, specifies the number of color components.
     * @param width width of the texture.
     * @param height height of the texture.
     * 
     * The difference with last constructor is the usage of glTexStorage2D
     */    
    CanvasCUDA(GLenum target, GLenum internalFormat, GLsizei width, 
        GLsizei height, unsigned int flags=cudaGraphicsRegisterFlagsSurfaceLoadStore);

    /**
     * @brief Copy operator.
     * 
     * Copy operator is deleted, doesn't allow copy.
     */
    CanvasCUDA( const CanvasCUDA& ) = delete;

    /**
     * @brief Asignment by copy operator.
     * 
     * Asignment by copy operator is deleted, doesn't allow copy.
     */
    CanvasCUDA& operator=( const CanvasCUDA& ) = delete;
    
    /**
     * @brief Movement constructor.
     * 
     * Allows object movement.
     */
    CanvasCUDA( CanvasCUDA&& ) noexcept;

    /**
     * @brief Movement by asignment constructor.
     * 
     * Allows object movement by asignment.
     */
    CanvasCUDA& operator=( CanvasCUDA&& ) noexcept;

    /**
     * @brief Class default constructor.
     * 
     * Class destructor. No manually deallocation needed.
     */
    ~CanvasCUDA() = default;

    /**
     * @brief Get canvas texture.
     * 
     * Gets canvas texture object reference.
     * 
     * @return Canvas texture object reference.
     */
    Texture& getTexture();
    
    
    /**
     * @brief Get canvas texture's CUDA wrapper .
     * 
     * Gets canvas texture object reference.
     * 
     * @return Canvas texture object reference.
     */
    TextureCUDAWrapper& getCUDATextureWrapper();


    DepthDataCUDA& getDepthDataCUDA();

    DepthDownsampleCUDA& getDepthDownsampleDataCUDA();

    /**
     * @brief Get canvas framebuffer.
     * 
     * Gets canvas framebuffer object reference.
     * 
     * @return Canvas framebuffer object reference.
     */
    Framebuffer& getFramebuffer();

    /**
     * @brief Get canvas width.
     * 
     * Gets canvas (thus framebuffer and texture) width.
     * 
     * @return canvas width.
     */
    GLsizei getWidth() const;

    /**
     * @brief Get canvas height.
     * 
     * Gets canvas (thus framebuffer and texture) height.
     * 
     * @return canvas height.
     */
    GLsizei getHeight() const;

    /**
     * @brief Texture late setting.
     * 
     * Texture generation, binding and setup. Generates a texture with a buffer target, a pixel
     * format, width and height and data type. 
     * 
     * @param target buffer target to set.
     * @param internalFormat image format, specifies the number of color components.
     * @param width width of the texture.
     * @param height height of the texture.
     * @param format format of the pixel data.
     * @param type data type of the pixel. 
     */
    void setTexture(GLenum target, GLenum internalFormat, GLsizei width, 
        GLsizei height, GLenum format, GLenum type);

    void setCUDATextureWrapper(GLenum target, unsigned int flags=cudaGraphicsRegisterFlagsSurfaceLoadStore);

    /**
     * @brief Texture image late setting.
     * 
     * Texture image generation, binding and setup. Generates a texture image with a buffer target, a pixel
     * format, width and height. 
     * 
     * @param target buffer target to set.
     * @param internalFormat image format, specifies the number of color components.
     * @param width width of the texture.
     * @param height height of the texture.
     */
    void setTextureImage(GLenum target, GLenum internalFormat, GLsizei width, 
        GLsizei height);

    /**
     * @brief Generate and late setup of framebuffer.
     * 
     * Generate and late setup of framebuffer with default framebuffer target, the canvas texture and
     * a given attachment.
     * 
     * @param attachment attachment point to which an image form texture should be attached.
     */
    void setFBO(GLenum attachment);

    /**
     * @brief Binds canvas texture.
     * 
     * Binds canvas texture to its specified unit.
     */
    void bindTexture() const;

    /**
     * @brief Binds canvas texture image.
     * 
     * Binds canvas texture image to specified unit.
     * 
     * @param access access types to image from shaders: GL_READ_ONLY, GL_WRITE_ONLY, or GL_READ_WRITE.
     * @param format pixel format used for image formatted stores on shaders.
     */
    void bindTextureImage(GLenum access, GLenum format) const;

    /**
     * @brief Bind canvas framebuffer.
     */
    void bindFBO() const;

    /**
     * @brief Takes screenshot of texture.
     * 
     * Takes screenshot of texture by reading framebuffer data. It's stored on screenshot_file.
     * 
     * @param screenshot_file name of the image to store.
     */
    void takeScreenshot(std::string screenshot_file) const;

    /**
     * @brief sets up the texture used as depth storage and the depth buffer used in
     * code to apply atomics to it.
     * 
     * @param width width of the texture (depends on the screen resolution).
     * @param height height of the texture (depends on the screen resolution).
     */
    void setupDepthDataCUDA(int width, int height);

    
    /**
     * * @brief sets up a depth downsampled texture for mipmap generation on a customized
     * shader program.
     * 
     * @param desiredLevel desired mipmap level of resolution.
     * @param width width of the texture (depends on the screen resolution).
     * @param height height of the texture (depends on the screen resolution).
     */
    void setupDepthDownsampleCUDA(int desiredLevel, int width, int height);
    

    /**
     * @brief cleans the canvas framebuffer, depth buffers and customized using a compute shader.
     * 
     * @param workGroupSizeXPerPixel number of work groups per pixel in X direction.
     * @param workGroupSizeYPerPixel number of work groups per pixel in Y direction.
     * @param far far plane distance (restart value in depth buffer).
     */
    // void cleanCanvasBuffers(int workGroupSizeXPerPixel, int workGroupSizeYPerPixel, float far);

    /**
     * @brief downsampled depth texture mipmap generation. Dispatches the program to compute the
     * downsampled depth texture mipmap.
     * 
     * @param downsampleWorkGroupSizeX number of work groups per pixel in X direction.
     * @param downsampleWorkGroupSizeY number of work groups per pixel in Y direction.
     */
    // void downsampleCanvasDepth(int downsampleWorkGroupSizeX, int downsampleWorkGroupSizeY);

private:
    Texture m_canvas; ///< texture used as a canvas to draw pixels.
    TextureCUDAWrapper m_canvasTexWrapper; ///< texture used as a canvas to draw pixels.
    
    Framebuffer m_fbo; ///< framebuffer object associated to texture.
    GLsizei m_width; ///< width of the canvas.
    GLsizei m_height; ///< height of the canvas.
    
    DepthDataCUDA m_depthData; ///< depth data object associated to canvas.
    DepthDownsampleCUDA m_depthDownsample; ///< depth downsample object associated to canvas.

    bool m_isCleaningProgramSetup = false; ///< canvas cleaning program setup flag.
    bool m_isDownsampleProgramSetup = false; ///< depth downsample program setup flag.
    bool m_isDepthDataSetup = false; ///< depth data setup flag.
    bool m_isDepthDownsampleSetup = false; ///< depth downsample setup flag.
};

#endif