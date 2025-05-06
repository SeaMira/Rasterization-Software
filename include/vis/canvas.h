#ifndef _CANVAS_H_
#define _CANVAS_H_

#include <SDL3/SDL.h>
#include "vis/compute_shader_program.h"
#include "vis/gl/frame_buffer.h"
#include "vis/gl/texture.h"
#include "vis/gl/depth_data.h"
#include "vis/gl/depth_downsample.h"

/**
 * @class Canvas 
 * 
 * @brief Canvas class used to set the viewport in which to draw directly.
 * 
 * Used as direct canvas to draw and show in the viewport. It's texture and framebuffer
 * are the main ones to be shown.
 */
class Canvas
{
public:
    /**
     * @brief Class default constructor.
     * 
     * Class default constructor for future setup.
     */
    Canvas() = default;

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
    Canvas(GLenum target, GLenum internalFormat, GLsizei width, 
        GLsizei height, GLenum format, GLenum type);

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
    Canvas(GLenum target, GLenum internalFormat, GLsizei width, 
        GLsizei height);

    /**
     * @brief Copy operator.
     * 
     * Copy operator is deleted, doesn't allow copy.
     */
    Canvas( const Canvas& ) = delete;

    /**
     * @brief Asignment by copy operator.
     * 
     * Asignment by copy operator is deleted, doesn't allow copy.
     */
    Canvas& operator=( const Canvas& ) = delete;
    
    /**
     * @brief Movement constructor.
     * 
     * Allows object movement.
     */
    Canvas( Canvas&& ) noexcept;

    /**
     * @brief Movement by asignment constructor.
     * 
     * Allows object movement by asignment.
     */
    Canvas& operator=( Canvas&& ) noexcept;

    /**
     * @brief Class default constructor.
     * 
     * Class destructor. No manually deallocation needed.
     */
    ~Canvas() = default;

    /**
     * @brief Get canvas texture.
     * 
     * Gets canvas texture object reference.
     * 
     * @return Canvas texture object reference.
     */
    Texture& getTexture();

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
    void setupDepthData(int width, int height);

    /**
     * @brief sets up a "cleaning" shader program. Restart the depth buffer and the
     * depth texture values.
     * 
     * Restart values on framebuffer, depth buffer and depth texture. May also be used to restart values on 
     * other buffers (pixel ownership buffer, for example).
     * 
     * @param cleaningShader compute shader used to clean desired buffers.
     */
    void setupCleaningProgram(ComputeShader& cleaningShader);
    
    /**
     * * @brief sets up a depth downsampled texture for mipmap generation on a customized
     * shader program.
     * 
     * @param desiredLevel desired mipmap level of resolution.
     * @param width width of the texture (depends on the screen resolution).
     * @param height height of the texture (depends on the screen resolution).
     */
    void setupDepthDownsample(int desiredLevel, int width, int height);
    
    /**
     * @brief generates a downsampled depth texture mipmap.
     * 
     * @param downsampleShader compute shader used to generate downsampled texture.  
     */
    void setupDownsamplingProgram(ComputeShader& downsampleShader);

    /**
     * @brief cleans the canvas framebuffer, depth buffers and customized using a compute shader.
     * 
     * @param workGroupSizeXPerPixel number of work groups per pixel in X direction.
     * @param workGroupSizeYPerPixel number of work groups per pixel in Y direction.
     * @param far far plane distance (restart value in depth buffer).
     */
    void cleanCanvasBuffers(int workGroupSizeXPerPixel, int workGroupSizeYPerPixel, float far);

    /**
     * @brief downsampled depth texture mipmap generation. Dispatches the program to compute the
     * downsampled depth texture mipmap.
     * 
     * @param downsampleWorkGroupSizeX number of work groups per pixel in X direction.
     * @param downsampleWorkGroupSizeY number of work groups per pixel in Y direction.
     */
    void downsampleCanvasDepth(int downsampleWorkGroupSizeX, int downsampleWorkGroupSizeY);

private:
    Texture m_canvas; ///< texture used as a canvas to draw pixels.
    Framebuffer m_fbo; ///< framebuffer object associated to texture.
    GLsizei m_width; ///< width of the canvas.
    GLsizei m_height; ///< height of the canvas.
    
    DepthData m_depthData; ///< depth data object associated to canvas.
    DepthDownsample m_depthDownsample; ///< depth downsample object associated to canvas.

    ComputeShader* m_cleaningProgram; ///< compute shader used to clean desired buffers.
    ComputeShader* m_downsampleProgram; ///< compute shader used to generate mipmap levels.

    bool m_isCleaningProgramSetup = false; ///< canvas cleaning program setup flag.
    bool m_isDownsampleProgramSetup = false; ///< depth downsample program setup flag.
    bool m_isDepthDataSetup = false; ///< depth data setup flag.
    bool m_isDepthDownsampleSetup = false; ///< depth downsample setup flag.
};

#endif