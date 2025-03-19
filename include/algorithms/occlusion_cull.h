#ifndef _OCCLUSION_CULLING_H_
#define _OCCLUSION_CULLING_H_

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>

struct PixelZ
{
    int x, y;
    float z;
    PixelZ(int x, int y, float z) : x(x), y(y), z(z) {}
};

struct DepthBuffer 
{
    int m_width, m_height;
    float m_maxDepth;
    std::vector<float> data;
    
    
    float getDepth(int x, int y) const 
    {
        if (x < 0 || y < 0 || x >= m_width || y >= m_height)
        {
            // std::cout << "depth index out of bounds" << std::endl;
            return m_maxDepth;  
        }
        return data[(m_height - y - 1) * m_width + x];
    }
};

using HierarchicalZBuffer = std::vector<DepthBuffer>;

HierarchicalZBuffer generateHiZPyramid(const DepthBuffer& depthBuffer, int maxLevels);

bool isPixelOccluded(float pixelDepth, const HierarchicalZBuffer& hizPyramid, int screenX, int screenY);
bool isSphereBillboardVisible(std::vector<PixelZ>& spherePixels, const HierarchicalZBuffer& hizPyramid);
#endif