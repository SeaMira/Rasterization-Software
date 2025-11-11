#ifndef _DEPTH_DOWNSAMPLE_CU_
#define _DEPTH_DOWNSAMPLE_CU_

#include <glad/glad.h>
#include <iostream>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "utils/parallel/cuda_checks.h"

/**
 * @class DepthDownsampleCUDA
 * 
 * @brief Depth downsample for occlusion culling on GPU.
 * 
 * Downsampled depth data texture. Based on a "just desired level construction" approach
 */
class DepthDownsampleCUDA
{
public:

    /**
     * @brief Default constructor
     * 
     * Default constructor for future setup.
     */
    DepthDownsampleCUDA();

    /**
     * @brief Class constructor
     * 
     * Generates a texture storage that will work as the depth downsample buffer texture.
     * 
     * @param desiredLevel desired mipmap level of resolution.
     * @param width width of the texture.
     * @param height height of the texture.
     */
    DepthDownsampleCUDA(int desiredLevel, int width, int height);

    /**
     * @brief Class destructor
     * 
     * Deletes the texture buffer, freeing space.
     */
    ~DepthDownsampleCUDA();

    /**
     * @brief Setup function
     * 
     * Sets the depth downsample texture with the desired mipmap level of resolution.
     * 
     * @param desiredLevel desired mipmap level of resolution.
     * @param width width of the texture.
     * @param height height of the texture.
     */
    void setup(int desiredLevel, int width, int height);
    
    /**
     * @brief Setup function
     * 
     * Sets the depth downsample texture with the desired mipmap level of resolution.
     */
    void setup();

    /**
     * @brief Binds pyramid's texture in the context as sampler (just read). 
     */
    void resolutionSetup(int width, int height);
    
    /**
     * @brief Binds pyramid's texture in the context as sampler (just read). 
     */
    void downsamplingLevelSetup(int desiredLevel);

    /**
     * @brief Unbinds pyramid's texture (as sampler) in the context. 
     */
    void memoryAlloc();
    
    /**
     * @brief Binds pyramid's texture buffer as image2D in the context (read/write). 
     */
    void writingSurfaceSetup();

    /**
     * @brief Unbinds pyramid's texture (as image2D) in the context. 
     */
    void readingSurfaceSetup();

    bool isValid() const;

    cudaSurfaceObject_t getSurface() const;

    cudaTextureObject_t getTexture() const;


private:
    int m_desiredLevel; ///< Desired mipmap level of resolution.
    int m_width; ///< Screen resolution width to downsample.
    int m_height; ///< Screen resolution height to downsample.

    cudaArray_t m_downsampleArray;        ///< Textura destino (mipmap/<)

    cudaResourceDesc m_surfResDesc = {};
    cudaTextureDesc m_texDesc = {};

    cudaSurfaceObject_t m_surface;        ///< Para escritura (downsample)
    cudaTextureObject_t m_texture;        ///< Para lectura (otros kernels
};

#endif // _DEPTH_DOWNSAMPLE_CU_