#include "algorithms/occlusion_cull.h"

std::vector<DepthBuffer> generateHiZPyramid(const DepthBuffer& depthBuffer, int maxLevels) 
{
    std::vector<DepthBuffer> pyramid;

    // Add first level (full resolution)
    pyramid.push_back(depthBuffer);

    int currentWidth = depthBuffer.m_width;
    int currentHeight = depthBuffer.m_height;

    // Mipmap level generation
    int level = 0;
    while ( level < maxLevels || (currentWidth > 1 || currentHeight > 1)) 
    {
        currentWidth = std::max(1, currentWidth / 2);
        currentHeight = std::max(1, currentHeight / 2);

        DepthBuffer mipLevel;
        mipLevel.m_width = currentWidth;
        mipLevel.m_height = currentHeight;
        mipLevel.data.resize(currentWidth * currentHeight);

        const DepthBuffer& prevLevel = pyramid.back();

        // Every mipmap texel saves de greatest depth from 2x2 last level texels
        for (int y = 0; y < currentHeight; y++) 
        {
            for (int x = 0; x < currentWidth; x++) 
            {
                float d0 = prevLevel.getDepth(x * 2, y * 2);
                float d1 = prevLevel.getDepth(x * 2 + 1, y * 2);
                float d2 = prevLevel.getDepth(x * 2, y * 2 + 1);
                float d3 = prevLevel.getDepth(x * 2 + 1, y * 2 + 1);

                // Save max depth from new 2x2 texel
                mipLevel.data[(currentHeight - y - 1) * currentWidth + x] = std::max({d0, d1, d2, d3});
            }
        }
        level++;
        pyramid.push_back(mipLevel);
    }

    return pyramid;
}


bool isPixelOccluded(float pixelDepth, const HierarchicalZBuffer& hiZPyramid, int screenX, int screenY)
{
    for (int level = (hiZPyramid.size() - 1); level >= 0; level--)
    {
        int mipWidth = hiZPyramid[level].m_width;
        int mipHeight = hiZPyramid[level].m_height;

        int mipX = static_cast<int>((float)(screenX * mipWidth) / (float)hiZPyramid[0].m_width);
        int mipY = static_cast<int>((float)(screenY * mipHeight) / (float)hiZPyramid[0].m_height);

        float hizDepth = hiZPyramid[level].getDepth(mipX, mipY);

        if (pixelDepth > hizDepth) 
            return true;  // Sphere occluded
          
    }

    return false;  // Sphere not occluded
}

bool isSphereBillboardVisible(std::vector<PixelZ>& spherePixels, const HierarchicalZBuffer& hiZPyramid)
{
    std::vector<float> depths;
    for (const PixelZ& pixel : spherePixels)
    {
        float currentDepth;
        for (int level = (hiZPyramid.size() - 1); level >= 0; level--)
        {
            int mipWidth = hiZPyramid[level].m_width;
            int mipHeight = hiZPyramid[level].m_height;

            int mipX = static_cast<int>((float)(pixel.x * mipWidth) / (float)hiZPyramid[0].m_width);
            int mipY = static_cast<int>((float)(pixel.y * mipHeight) / (float)hiZPyramid[0].m_height);

            currentDepth = hiZPyramid[level].getDepth(mipX, mipY);

            
            if (pixel.z > currentDepth || level == 0) 
                depths.push_back(currentDepth);
        }
    }
    for (const float& depth : depths)
        for (const PixelZ& pixel : spherePixels)
            // TODO: check delta depth depending on camera speed/framerate? 
            if (pixel.z <= depth)
                return true;
        
    return false;

}