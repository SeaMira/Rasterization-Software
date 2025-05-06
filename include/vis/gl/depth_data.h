#ifndef _DEPTH_DATA_
#define _DEPTH_DATA_

#include <glad/glad.h>
#include <iostream>
#include <stdexcept>
#include "vis/gl/texture.h"
#include "vis/gl/storage_buffer.h"

class DepthData
{
public:

    /**
     * @brief Default constructor
     * 
     * Default constructor for future setup.
     */
    DepthData() = default;

    /**
     * @brief Class constructor of the DepthData class. Gets necessary data for the depth texture
     * and depth buffer to store the future depth data. Sets the format and access to them, besides the 
     * gl binding points for both.
     * 
     * @param width width of the texture.
     * @param height height of the texture.
     * @param textureBindingPoint gl texture binding point.
     * @param bufferBindingPoint gl buffer binding point.
     */ 
    DepthData(int width, int height, int textureBindingPoint, int bufferBindingPoint);
    
    /**
     * @brief Default class destructor.
     */
    ~DepthData() = default;

    /**
     * @brief Setup function for late retrieving of the depth data settings.
     */
    void setup(int width, int height, int textureBindingPoint, int bufferBindingPoint);

    /**
     * @brief Binds the texture to the context as an image2D (read/write).
     */
    void bindTexture();

    /**
     * @brief Unbinds the texture (quit image2D).
     */
    void unbindTexture();

    /**
     * @brief Binds the texture to the context as a sampler (just read).
     */
    void bindTextureImage(GLenum access, GLenum format);

    /**
     * @brief Unbinds the texture (quit sampler).
     */
    void unbindTextureImage(GLenum access, GLenum format);
    
    /**
     * @brief Binds the storage buffer to the context as a SSBO.
     */
    void bindStorageBuffer();

    /**
     * @brief Unbinds the SSBO.
     */
    void unbindStorageBuffer();

private:
    int m_screenWidth; ///< width of the screen resolution.
    int m_screenHeight; ///< height of the screen resolution
    int m_textureBindingPoint; ///< gl binding point for the texture.
    int m_bufferBindingPoint; ///< gl binding point for the buffer.
    Texture m_texture; ///< Gl texture object.
    StorageBuffer m_storageBuffer; ///< Gl storage buffer object.
};

#endif // _DEPTH_DATA_