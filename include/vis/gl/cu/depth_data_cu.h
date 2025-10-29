#ifndef _DEPTH_DATA_CU_
#define _DEPTH_DATA_CU_

#include <glad/glad.h>
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

class DepthDataCUDA
{
public:

    /**
     * @brief Default constructor
     * 
     * Default constructor for future setup.
     */
    DepthDataCUDA() = default;

    /**
     * @brief Class constructor of the DepthDataCUDA class. Gets necessary data for the depth texture
     * and depth buffer to store the future depth data. Sets the format and access to them, besides the 
     * gl binding points for both.
     * 
     * @param width width of the texture.
     * @param height height of the texture.
     */ 
    DepthDataCUDA(int width, int height);
    
    /**
     * @brief Class destructor.
     * 
     * Frees the allocated CUDA memory.
     */
    ~DepthDataCUDA();

    /**
     * @brief Setup resolution of the depth buffer
     * 
     * This function sets up the resolution of the depth buffer.
     * 
     * @param width width of the texture.
     * @param height height of the texture.
     */
    void resolutionSetup(int width, int height);

    /**
     * @brief Reserves space for CUDA array.
     */
    void memoryAlloc();

    /**
     * @brief Unbinds the texture (quit image2D).
     */
    void setup(int width, int height);

    /**
     * @brief Returns pointer to depth buffer.
     */
    unsigned int* getDepthBuffer() const;

    /**
     * @brief Returns screen width.
     */
    int getScreenWidth() const;

    /**
     * @brief Returns screen height.
     */
    int getScreenHeight() const;

private:
    int m_screenWidth; ///< width of the screen resolution.
    int m_screenHeight; ///< height of the screen resolution
    unsigned int *m_depthBuffer; ///< gl binding point for the texture.
    
};

#endif // _DEPTH_DATA_CU_