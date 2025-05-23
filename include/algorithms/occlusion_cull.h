#ifndef _OCCLUSION_CULLING_H_
#define _OCCLUSION_CULLING_H_

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>


/**
 * @struct PixelZ
 * @brief Represents a pixel with its coordinates and depth on projection space.
 *
 */
struct PixelZ
{
    int x; /**< Coordinate along the width of the viewport */
    int y; /**< Coordinate along the height of the viewport */
    float z; /**< Depth associated to the projection of this pixel on the z-axis */

    /**
     * @brief Construct a new PixelZ object
     * 
     * @param x Coordinate along the width of the viewport
     * @param y Coordinate along the height of the viewport
     * @param z Depth associated to the projection of this pixel on the z-axis
     */
    PixelZ(int x, int y, float z) : x(x), y(y), z(z) {}
};

/**
 * @struct DepthBuffer
 * @brief Represents a depth buffer. 
 * 
 * Handles the depth values of the pixels on the screen. Uses the default cpp constructor.
 *
 */
struct DepthBuffer 
{
    int m_width; /**< Width of the depth buffer */
    int m_height; /**< Height of the depth buffer */
    float m_maxDepth; /**< Maximum depth value achievable*/
    std::vector<float> data; /**< Depth values of the pixels */
    
    /**
     * @brief Gets the depth value of a pixel on the depth buffer.
     * 
     * By giving a coordinate (pixel position en x/y axis) it returns the depth value of the pixel. If coordinates
     * given are out of bounds, it returns the maximum depth value.
     * 
     * @param x The x coordinate of the pixel.
     * @param y The y coordinate of the pixel.
     * 
     * @return The depth value of the pixel.
     */
    float getDepth(int x, int y) const 
    {
        if (x < 0 || y < 0 || x >= m_width || y >= m_height)
        {
            // std::cout << "depth index out of bounds" << std::endl;
            return m_maxDepth;  
        }
        return data[(m_height - y - 1) * m_width + x];
    }

    /**
     * @brief Gets the maximum depth value stored in the depth buffer.
     * 
     * @return The maximum depth value stored in the depth buffer.
     */
    float getMaxDepthStored()
    {
        float max = 0.0f;
        for (float& depth: data)
        {
            if (depth < 1.0f)
            {
                if (depth > max || max > 1.0f) max = depth;
            }
        }
        if (max == 0.0f) return 0.001f;
        return max;
    }

    /**
     * @brief Gets the minimum depth value stored in the depth buffer.
     * 
     * @return The minimum depth value stored in the depth buffer.
     */
    float getMinDepthStored()
    {
        float min = 1.0f;
        for (float& depth: data)
        {
            if (depth < 1.0f)
            {
                if (depth < min) min = depth;
            }
        }
        if (min == 0.0f) return 0.001f;
        return min;
    }
};

/**
 * @brief Represents a Hierarchical Z Buffer.
 * 
 * A Hierarchical Z Buffer is a pyramid of depth buffers. Each level of the pyramid has a smaller resolution
 * than the previous one. The first level is the original depth buffer (full resolution).
 * 
 */
using HierarchicalZBuffer = std::vector<DepthBuffer>;

/**
 * @brief Generates a Hierarchical Z Buffer pyramid from a depth buffer.
 * 
 * Generates a Hierarchical Z Buffer pyramid from a depth buffer. Each upper level is half the resolution of the last one so the first level is full resolution,
 * the second level has texels of size 2x2 pixels, the 3rd level texels have 4x4 pixels... Each texel stores the max depth of the pixels it represents.
 *  The number of levels of the pyramid is given by the parameter maxLevels.
 * 
 * @param depthBuffer The full resolution depth buffer.
 * @param maxLevels The number of levels of the pyramid.
 * 
 * @return The Hierarchical Z Buffer pyramid.
 */
HierarchicalZBuffer generateHiZPyramid(const DepthBuffer& depthBuffer, int maxLevels);

/**
 * @brief Tells if a pixel projection's depth is occluded by pixels on the Hierarchical Z Buffer pyramid.
 * 
 * Given the coordinates of a pixel and its depth, it tells if the pixel is occluded by other pixels on the Hierarchical Z Buffer pyramid in the same place.
 * Checks different levels of the pyramid to determine if the pixel is occluded.
 * 
 * @param pixelDepth A float representing the depth to be tested
 * @param hizPyramid The HierarchicalZBuffer representing the different depth precision levels.
 * @param screenX The x coordinate of the pixel on the screen.
 * @param screenY The y coordinate of the pixel on the screen.
 * 
 * @return The Hierarchical Z Buffer pyramid.
 */
bool isPixelOccluded(float pixelDepth, const HierarchicalZBuffer& hizPyramid, int screenX, int screenY);

/**
 * @brief Tells if a pixel projection's depth is occluded by pixels on the Hierarchical Z Buffer pyramid.
 * 
 * Given the coordinates of a pixel and its depth, it tells if the pixel is occluded by other pixels on the Hierarchical Z Buffer pyramid in the same place.
 * Checks last level of the pyramid to determine if the pixel is occluded.
 * 
 * @param pixelDepth A float representing the depth to be tested
 * @param hizPyramid The HierarchicalZBuffer representing the different depth precision levels.
 * @param screenX The x coordinate of the pixel on the screen.
 * @param screenY The y coordinate of the pixel on the screen.
 * 
 * @return The Hierarchical Z Buffer pyramid.
 */
bool isPixelLastLevelOccluded(float pixelDepth, const HierarchicalZBuffer& hizPyramid, int screenX, int screenY);

/**
 * @brief Tells if certain pixels of a billboard are occluded by pixels on the Hierarchical Z Buffer pyramid.
 * 
 * Given the pixels projected from a billboard and the Hierarchical Z Buffer pyramid, it tells if the pixels are occluded by other pixels on the Hierarchical Z Buffer pyramid.
 *
 * @param pixels A vector of PixelZ representing the pixels of the billboard.
 * @param hizPyramid The HierarchicalZBuffer representing the different depth precision levels.
 * 
 * @return true if the billboard is visible, false otherwise. 
 */
bool isBillboardVisible(std::vector<PixelZ>& pixels, const HierarchicalZBuffer& hizPyramid);
#endif