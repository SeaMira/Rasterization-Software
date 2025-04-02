#ifndef _DEPTH_PYRAMID_
#define _DEPTH_PYRAMID_

#include <glad/glad.h>
#include <iostream>
#include <stdexcept>

/**
 * @class DepthPyramid
 * 
 * @brief Depth pyramid for occlusion culling.
 * 
 * Hierarchical Z pyramid made of depth mipmaps levels. Used for occlusion culling.
 * Every level has texels half the resolution of the last one, the base being the full
 * depth resolution buffer.
 */
class DepthPyramid
{
public:

    /**
     * @brief Class constructor
     * 
     * Generates a texture storage that will work as the depth pyramid buffer
     * 
     * @param maxLevels maximum pyramid level of resolution.
     * @param width width of the texture.
     * @param height height of the texture.
     */
    DepthPyramid(int maxLevels, int width, int height);

    /**
     * @brief Class destructor
     * 
     * Deletes the texture buffer, freeing space.
     */
    ~DepthPyramid();

    /**
     * @brief Binds pyramid's texture buffer level in the context.
     * 
     * Binds the texture buffer for writing.
     * 
     * @param level mipmap level we are going to bind.
     */
    void bindLevel(int bindingPoint, int level, GLenum access) const;
    

    /**
     * @brief Binds pyramid's texture buffer in the context. 
     */
    void bindTexture() const;

    /**
     * @brief Unbinds pyramid's texture buffer in the context. 
     */
    void unbindTexture() const;


private:
    GLuint m_hiZTexture; ///< Gl texture ID.
    int m_maxLevels; ///< Pyiramid levels amount.
    int m_width; ///< Pyramid texture width.
    int m_height; ///< Pyramid texture height.
};

#endif // _DEPTH_PYRAMID_