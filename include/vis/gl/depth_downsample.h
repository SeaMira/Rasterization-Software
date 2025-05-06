#ifndef _DEPTH_DOWNSAMPLE_
#define _DEPTH_DOWNSAMPLE_

#include <glad/glad.h>
#include <iostream>
#include <stdexcept>
#include "vis/gl/depth_data.h"

/**
 * @class DepthDownsample
 * 
 * @brief Depth downsample for occlusion culling on GPU.
 * 
 * Downsampled depth data texture. Based on a "just desired level construction" approach
 */
class DepthDownsample
{
public:

    /**
     * @brief Default constructor
     * 
     * Default constructor for future setup.
     */
    DepthDownsample() = default;

    /**
     * @brief Class constructor
     * 
     * Generates a texture storage that will work as the depth downsample buffer texture.
     * 
     * @param desiredLevel desired mipmap level of resolution.
     * @param width width of the texture.
     * @param height height of the texture.
     * @param bindingPoint gl texture binding point.
     */
    DepthDownsample(int desiredLevel, int width, int height, int bindingPoint);

    /**
     * @brief Class destructor
     * 
     * Deletes the texture buffer, freeing space.
     */
    ~DepthDownsample() = default;

    /**
     * @brief Setup function
     * 
     * Sets the depth downsample texture with the desired mipmap level of resolution.
     * 
     * @param desiredLevel desired mipmap level of resolution.
     * @param width width of the texture.
     * @param height height of the texture.
     * @param bindingPoint gl texture binding point.
     */
    void setup(int desiredLevel, int width, int height, int bindingPoint);

    /**
     * @brief Binds pyramid's texture in the context as sampler (just read). 
     */
    void bindImage(GLenum access, GLenum format) const;

    /**
     * @brief Unbinds pyramid's texture (as sampler) in the context. 
     */
    void unbindImage(GLenum access, GLenum format) const;
    
    /**
     * @brief Binds pyramid's texture buffer as image2D in the context (read/write). 
     */
    void bind() const;

    /**
     * @brief Unbinds pyramid's texture (as image2D) in the context. 
     */
    void unbind() const;


private:
    Texture m_downSampledTexture; ///< Gl texture ID.
    int m_desiredLevel; ///< Desired mipmap level of resolution.
    int m_bindingPoint; ///< gl texture binding point.
    int m_width; ///< Screen resolution width to downsample.
    int m_height; ///< Screen resolution height to downsample.
};

#endif // _DEPTH_DOWNSAMPLE_