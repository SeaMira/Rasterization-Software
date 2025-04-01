#include <iostream>
#include "utils/sequential/aux_functions.h"


float iSphere(glm::vec3 ro, glm::vec3 rd, glm::vec3 sph, float radius )
{
    float a = glm::dot(rd, rd);
    float b = glm::dot( rd, sph );
    float c = glm::dot( sph, sph ) - radius*radius;
    float h = b*b - a * c;
    if( h < 0.0f || a == 0.0f) return -1.0f;
    return (b - std::sqrt( h ))/a;
}


bboxCorners getSphereBbox(const glm::vec3& cameraSpaceSphere, const glm::vec3& camImposPos, 
    const glm::vec3& normCamSpaceSphere, const glm::mat4& proj, const glm::vec3& camPos, 
    const glm::vec3& front, const glm::vec3& up,const float sphRadius)
{
    const float sinAngle = sphRadius / (glm::length(cameraSpaceSphere) + 1e-6f);
    const float tanAngle = std::tan(std::asin(sinAngle));
    const float quadScale = tanAngle * glm::length(camImposPos);

    glm::vec3 impU = glm::normalize(glm::cross(normCamSpaceSphere, up));
    glm::vec3 impV = glm::cross(impU, normCamSpaceSphere) * quadScale;
    impU *= quadScale;

    const glm::vec3 upRightCorner = camImposPos + impU + impV;
    const glm::vec3 upLeftCorner = camImposPos - impU + impV;
    const glm::vec3 downRightCorner = camImposPos + impU - impV;
    const glm::vec3 downLeftCorner = camImposPos - impU - impV;

    const glm::vec4 upRight = proj * glm::vec4(upRightCorner, 1.0f);
    const glm::vec4 upLeft = proj * glm::vec4(upLeftCorner, 1.0f);
    const glm::vec4 downRight = proj * glm::vec4(downRightCorner, 1.0f);
    const glm::vec4 downLeft = proj * glm::vec4(downLeftCorner, 1.0f);
    
    glm::vec3 ndcUpRight = glm::vec3(upRight.x / upRight.w, upRight.y / upRight.w, upRight.z / upRight.w);
    glm::vec3 ndcUpLeft = glm::vec3(upLeft.x / upLeft.w, upLeft.y / upLeft.w, upLeft.z / upLeft.w);
    glm::vec3 ndcDownRight = glm::vec3(downRight.x / downRight.w, downRight.y / downRight.w, downRight.z / downRight.w);
    glm::vec3 ndcDownLeft = glm::vec3(downLeft.x / downLeft.w, downLeft.y / downLeft.w, downLeft.z / downLeft.w);


    glm::vec2 minCorner = glm::min(glm::min(ndcUpRight, ndcUpLeft), glm::min(ndcDownRight, ndcDownLeft));
    glm::vec2 maxCorner = glm::max(glm::max(ndcUpRight, ndcUpLeft), glm::max(ndcDownRight, ndcDownLeft));


    return {upRightCorner, ndcUpRight.z, upLeftCorner, ndcUpLeft.z, downRightCorner, ndcDownRight.z, downLeftCorner, ndcDownLeft.z, minCorner, maxCorner}; 
}

uint32_t vecToColor(glm::vec3 lambertCos) 
{
    const uint8_t R = lambertCos.x;
    const uint8_t G = lambertCos.y;
    const uint8_t B = lambertCos.z;

    return (0xFF000000) | (R << 16) | (G << 8) | B;
}

bool drawSphere(const glm::mat4& proj, const glm::mat4& view, 
    const glm::vec3& up, const glm::vec3& front, const glm::vec3& camPos, 
    const int SCR_WIDTH, const int SCR_HEIGHT, 
    const glm::vec4& sphere,
    std::vector<uint32_t>& framebuffer, std::vector<float>& depthBuffer, 
    HierarchicalZBuffer& hizPyramid, 
    uint8_t& sphereVisibilityFrameCache, std::vector<int>& pixelOwnership, int& sphereIndex, int pixelsOwned)
{
    glm::vec3 cameraSpaceSphere = glm::vec3(view * glm::vec4(sphere[0], sphere[1], sphere[2], 1.0f));
    glm::vec3 normCamSpaceSphere = glm::normalize(cameraSpaceSphere);
    glm::vec3 camImposPos = cameraSpaceSphere - normCamSpaceSphere * sphere.w;

    bboxCorners sphereBbox = getSphereBbox(cameraSpaceSphere, camImposPos,
        normCamSpaceSphere, proj, camPos, front, up, sphere.w);
    
    glm::ivec2 screenMin, screenMax;

    screenMin.x = ((sphereBbox.minCorner.x) * 0.5f + 0.5f) * SCR_WIDTH;
    screenMin.y = (sphereBbox.minCorner.y * 0.5f + 0.5f) * SCR_HEIGHT;
    screenMax.x = ((sphereBbox.maxCorner.x) * 0.5f + 0.5f) * SCR_WIDTH;
    screenMax.y = (sphereBbox.maxCorner.y * 0.5f + 0.5f) * SCR_HEIGHT;

    const float difx = screenMax.x - screenMin.x;
    const float dify = screenMax.y - screenMin.y;
    
    if (difx*dify <= 2) return false;
    auto computeViewImpPos = [&](int px, int py) -> glm::vec3 {
        const float u = (float)(px - screenMin.x) / difx;
        const float v = (float)(py - screenMin.y) / dify;
        const glm::vec3 A = glm::mix(sphereBbox.upLeftCorner, sphereBbox.upRightCorner, u);
        const glm::vec3 B = glm::mix(sphereBbox.downLeftCorner, sphereBbox.downRightCorner, u);
        return glm::mix(A, B, v);
    };

    auto onSphDepth = [&](glm::vec3 viewImpPos) -> float {
        const float h = iSphere(camPos, viewImpPos, cameraSpaceSphere, sphere.w);
        const glm::vec3 hit = viewImpPos * h;
        const float depth = hit.z < 0.0f ? (hit.z * proj[2].z + proj[3].z) / -hit.z : FLT_MAX;
        return depth;
    };
    
    float mid_x = glm::min(glm::max((screenMin.x + screenMax.x)/2, 0), SCR_WIDTH);
    float mid_y = glm::min(glm::max((screenMin.y + screenMax.y)/2, 0), SCR_HEIGHT);
    glm::vec3 middleLeft = computeViewImpPos(mid_x-(int)(difx*0.49f), mid_y);
    glm::vec3 middleRight = computeViewImpPos(mid_x+(int)(difx*0.49f), mid_y);
    glm::vec3 middleBottom = computeViewImpPos(mid_x, mid_y-(int)(dify*0.49f));
    glm::vec3 middleTop = computeViewImpPos(mid_x, mid_y+(int)(dify*0.49f));
    glm::vec3 middlePixelsVec = computeViewImpPos((screenMin.x + screenMax.x) / 2, (screenMin.y + screenMax.y) / 2);
    
    std::vector<PixelZ> spherePixels = {
        PixelZ(mid_x-(int)(difx*0.49f), mid_y, onSphDepth(middleLeft)),
        PixelZ(mid_x+(int)(difx*0.49f), mid_y, onSphDepth(middleRight)),
        PixelZ(mid_x, mid_y-(int)(dify*0.49f), onSphDepth(middleBottom)),
        PixelZ(mid_x, mid_y+(int)(dify*0.49f), onSphDepth(middleTop)),
        PixelZ((screenMin.x + screenMax.x) / 2, (screenMin.y + screenMax.y) / 2, onSphDepth(middlePixelsVec))
    };
    // if (mid_x-(int)(difx*0.48f) >= 0 && mid_x-(int)(difx*0.48f) < SCR_WIDTH &&
    //     mid_y >= 0 && mid_y < SCR_HEIGHT)
    //     framebuffer[(SCR_HEIGHT - mid_y - 1) * SCR_WIDTH + (mid_x-(int)(difx*0.48f))] = 0xFFFFFFFF;
    // if (mid_x+(int)(difx*0.48f) >= 0 && mid_x+(int)(difx*0.48f) < SCR_WIDTH &&
    //     mid_y >= 0 && mid_y < SCR_HEIGHT)
    //     framebuffer[(SCR_HEIGHT - mid_y - 1) * SCR_WIDTH + (mid_x+(int)(difx*0.48f))] = 0xFFFFFFFF;
    // if (mid_x >= 0 && mid_x < SCR_WIDTH &&
    //     mid_y-(int)(dify*0.48f) >= 0 && mid_y-(int)(dify*0.48f) < SCR_HEIGHT)
    //     framebuffer[(SCR_HEIGHT - (mid_y-(int)(dify*0.48f)) - 1) * SCR_WIDTH + mid_x] = 0xFFFFFFFF;
    
    // if (mid_x >= 0 && mid_x < SCR_WIDTH &&
    //     mid_y+(int)(dify*0.48f) >= 0 && mid_y+(int)(dify*0.48f) < SCR_HEIGHT)
    //     framebuffer[(SCR_HEIGHT - (mid_y+(int)(dify*0.48f)) - 1) * SCR_WIDTH + mid_x] = 0xFFFFFFFF;
    
    if (!isSphereBillboardVisible(spherePixels, hizPyramid))
    {
        // std::cout << "pixels owned by: " << sphereIndex << " - " << pixelsOwned << " frames " << (int)(sphereVisibilityFrameCache & 0b01111111) << std::endl;
        if ((sphereVisibilityFrameCache & 0b01111111) == 0 && pixelsOwned == 0)
        {
            // std::cout << "no frames" << std::endl;
            return false;
        }
        else if ((sphereVisibilityFrameCache & 0b01111111) > 0 && pixelsOwned == 0)
            // std::cout << "menos frames de gracia para " << sphereIndex << std::endl;
            sphereVisibilityFrameCache = (sphereVisibilityFrameCache & 0b01111111) - 1;
    }     
    

    #pragma omp parallel for collapse(2)
    for (int px = std::max(0, screenMin.x); px < std::min(screenMax.x, SCR_WIDTH); px++)
    {
        bool finishedLine = false;
        for (int py = std::max(0, screenMin.y); py < std::min(SCR_HEIGHT, screenMax.y); py++)
        {

            const glm::vec3 viewImpPos = computeViewImpPos(px, py);
            const float h = iSphere(camPos, viewImpPos, cameraSpaceSphere, sphere.w);
            const int index = (SCR_HEIGHT - py - 1) * SCR_WIDTH + px;
            const bool showBbox = (px == screenMin.x || py == screenMin.y || px == screenMax.x - 1 || py == screenMax.y - 1);
            // if (showBbox) framebuffer[index] = vecToColor(glm::vec3(0));
            if ((h > 0.0f)) 
            {
                if (!finishedLine) finishedLine = true;
                const glm::vec3 hit = viewImpPos * h;
                const float depth = hit.z < 0.0f ? (hit.z * proj[2].z + proj[3].z) / -hit.z : FLT_MAX;
                if (depth < depthBuffer[index]) 
                {
                    const glm::vec3 normal = glm::normalize( hit - cameraSpaceSphere );
                    const float lambertCos = glm::dot(normal, -glm::normalize(hit));
                    depthBuffer[index] = depth;
                    framebuffer[index] = vecToColor(255 * lambertCos * lightColor * diffuseI);
                    pixelOwnership[index] = sphereIndex;
                }
            } else if (finishedLine)
            {
                break;
            }
        }
    }
    return true;
}


bool drawBillboard(const glm::mat4& proj, const glm::mat4& view, 
    const glm::vec3& up, const glm::vec3& front, const glm::vec3& camPos, 
    const int SCR_WIDTH, const int SCR_HEIGHT, 
    const glm::vec4& sphere,
    std::vector<uint32_t>& framebuffer, std::vector<float>& depthBuffer, HierarchicalZBuffer& hizPyramid)
{
    glm::vec3 cameraSpaceSphere = glm::vec3(view * glm::vec4(sphere[0], sphere[1], sphere[2], 1.0f));
    glm::vec3 normCamSpaceSphere = glm::normalize(cameraSpaceSphere);
    glm::vec3 camImposPos = cameraSpaceSphere - normCamSpaceSphere * sphere.w;

    bboxCorners sphereBbox = getSphereBbox(cameraSpaceSphere, camImposPos,
        normCamSpaceSphere, proj, camPos, front, up, sphere.w);
    
    glm::ivec2 screenMin, screenMax;

    screenMin.x = ((sphereBbox.minCorner.x) * 0.5f + 0.5f) * SCR_WIDTH;
    screenMin.y = (sphereBbox.minCorner.y * 0.5f + 0.5f) * SCR_HEIGHT;
    screenMax.x = ((sphereBbox.maxCorner.x) * 0.5f + 0.5f) * SCR_WIDTH;
    screenMax.y = (sphereBbox.maxCorner.y * 0.5f + 0.5f) * SCR_HEIGHT;

    const float difx = screenMax.x - screenMin.x;
    const float dify = screenMax.y - screenMin.y;
    
    bool dlC = isPixelOccluded(sphereBbox.downLeftCornerndcZ, hizPyramid, screenMin.x, screenMin.y);
    bool ulC = isPixelOccluded(sphereBbox.upLeftCornerndcZ, hizPyramid, screenMin.x, screenMax.y);
    bool drC = isPixelOccluded(sphereBbox.downRightCornerndcZ, hizPyramid, screenMax.x, screenMin.y);
    bool urC = isPixelOccluded(sphereBbox.upRightCornerndcZ, hizPyramid, screenMax.x, screenMax.y);
    if (dlC && ulC && drC && urC) return false;

    if (difx*dify <= 2) return false;
    #pragma omp parallel for collapse(2)
    for (int px = std::max(0, screenMin.x); px < std::min(screenMax.x, SCR_WIDTH); px++)
    {
        for (int py = std::max(0, screenMin.y); py < std::min(SCR_HEIGHT, screenMax.y); py++)
        {
            const int index = (SCR_HEIGHT - py - 1) * SCR_WIDTH + px;
            const bool showBbox = (px == screenMin.x || py == screenMin.y || px == screenMax.x - 1 || py == screenMax.y - 1);
            const float u = (float)(px - screenMin.x)/ difx;
            const float v = (float)(py - screenMin.y)/ dify;
            const float depthA = glm::mix(sphereBbox.upLeftCornerndcZ, sphereBbox.upRightCornerndcZ, u);
            const float depthB = glm::mix(sphereBbox.downLeftCornerndcZ, sphereBbox.downRightCornerndcZ, u);
            const float depth = glm::mix(depthA, depthB, v);
            // if (showBbox) framebuffer[index] = vecToColor(glm::vec3(0));
            if (depth < depthBuffer[index]) 
            {
                depthBuffer[index] = depth;
                if (showBbox) framebuffer[index] = vecToColor(glm::vec3(0));
                else framebuffer[index] = vecToColor(255.0f * glm::vec3(depth));
            }
        }
    }
    return true;
    
}


void drawMipmaps(HierarchicalZBuffer& hiZPyramid, int level,
    std::vector<uint32_t>& framebuffer,
    int SCR_WIDTH, int SCR_HEIGHT)
{
    float maxColor = hiZPyramid[level].getMaxDepthStored();
    float minColor = hiZPyramid[level].getMinDepthStored();
    for (int px = 0; px < SCR_WIDTH; px++)
    {
        for (int py = 0; py < SCR_HEIGHT; py++)
        {
            const int index = (SCR_HEIGHT - py - 1) * SCR_WIDTH + px;

            int mipWidth = hiZPyramid[level].m_width;
            int mipHeight = hiZPyramid[level].m_height;

            int mipX = static_cast<int>((float)(px * mipWidth) / (float)SCR_WIDTH);
            int mipY = static_cast<int>((float)(py * mipHeight) / (float)SCR_HEIGHT);
            float hizDepth = hiZPyramid[level].getDepth(mipX, mipY);
            float dif = (maxColor - minColor) == 0.0f ? 0.001 : (maxColor - minColor);
            float color = (hizDepth - minColor)/dif;
            framebuffer[index] = vecToColor(255.0f * glm::vec3(color));
        }
    } 
}