#include <iostream>
#include "utils/sequential/aux_functions_sphere.h"


float iSphere(glm::vec3 ro, glm::vec3 rd, glm::vec3 sph, float radius )
{
    glm::vec3 oc = ro - sph;
	float b = glm::dot( oc, rd );
	float c = glm::dot( oc, oc ) - radius*radius;
	float h = b*b - c;
	if( h<0.0 ) return -1.0;
	return -b - sqrt( h );

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

bool drawSphereWithOcclusionCulling(const glm::mat4& proj, const glm::mat4& view, 
    const glm::vec3& up, const glm::vec3& front, const glm::vec3& right, const glm::vec3& camPos, 
    const int SCR_WIDTH, const int SCR_HEIGHT, const float& fov, 
    const glm::vec4& sphere,
    std::vector<uint32_t>& framebuffer, std::vector<float>& depthBuffer, 
    HierarchicalZBuffer& hizPyramid, 
    uint8_t& sphereVisibilityFrameCache, std::vector<int>& pixelOwnership, int& sphereIndex, int pixelsOwned)
{
    glm::vec3 spherePos = glm::vec3(sphere[0], sphere[1], sphere[2]);
    glm::vec3 cameraSpaceSphere = glm::vec3(view * glm::vec4(spherePos, 1.0f));
    glm::vec3 normCamSpaceSphere = glm::normalize(cameraSpaceSphere);
    glm::vec3 camImposPos = cameraSpaceSphere - normCamSpaceSphere * sphere.w;

    bboxCorners sphereBbox = getSphereBbox(cameraSpaceSphere, camImposPos,
        normCamSpaceSphere, proj, camPos, front, up, sphere.w);
    
    float aspectRatio = (float)SCR_WIDTH / (float)SCR_HEIGHT;
    glm::ivec2 screenMin, screenMax;

    screenMin.x = ((sphereBbox.minCorner.x) * 0.5f + 0.5f) * SCR_WIDTH;
    screenMin.y = (sphereBbox.minCorner.y * 0.5f + 0.5f) * SCR_HEIGHT;
    screenMax.x = ((sphereBbox.maxCorner.x) * 0.5f + 0.5f) * SCR_WIDTH;
    screenMax.y = (sphereBbox.maxCorner.y * 0.5f + 0.5f) * SCR_HEIGHT;

    const float difx = screenMax.x - screenMin.x;
    const float dify = screenMax.y - screenMin.y;

    float fovRad = glm::radians(fov);
    float fovTan = tan(fovRad * 0.5f);
    float halfFovTan = fovTan * aspectRatio;

    ScreenRayCasting screenRayCasting(fov, aspectRatio, SCR_WIDTH, SCR_HEIGHT, right, up, front, view);

    // Delta per pixel
    glm::vec3 dx = screenRayCasting.dx;
    glm::vec3 dy = screenRayCasting.dy;

    // Start ray origin in world
    glm::vec3 rayStart = screenRayCasting.rayStart;
    
    if (difx*dify <= 2) return false;
    
    auto computeRd = [&](int px, int py) -> glm::vec3 {
        glm::vec2 p = (-glm::vec2(SCR_WIDTH, SCR_HEIGHT) + 2.0f*glm::vec2(px, py))/glm::vec2(SCR_WIDTH, SCR_HEIGHT);
        return glm::normalize( p.x*right* halfFovTan + p.y*up * fovTan + front );
    };

    auto onSphDepth = [&](glm::vec3 rd, int px, int py) -> float {
        const float h = iSphere(camPos, rd, spherePos, sphere.w);
        const glm::vec3 hit = (rayStart + float(px) * dx + float(py) * dy) * h;
        const float depth = (hit.z * proj[2].z + proj[3].z) / -hit.z;
        return depth;
    };

    // glm::vec3 sphToCam = glm::vec3(view * glm::vec4(spherePos - camPos, 1.0f));
    // glm::vec3 normsphToCam = glm::normalize(sphToCam);
    // glm::vec3 sphToCamImposPos = sphToCam - normsphToCam * sphere.w;
    
    // glm::vec4 centerProj = proj * glm::vec4(sphToCamImposPos, 1.0f);
    // glm::vec3 centerNdc = glm::vec3(centerProj.x / centerProj.w, centerProj.y / centerProj.w, centerProj.z / centerProj.w);
    // glm::vec3 centerRd = computeRd(((centerNdc.x) * 0.5f + 0.5f) * SCR_WIDTH, 
    //     (centerNdc.y * 0.5f + 0.5f) * SCR_HEIGHT);

    // const float centerDepth = onSphDepth(centerRd, ((centerNdc.x) * 0.5f + 0.5f) * SCR_WIDTH, 
    //     (centerNdc.y * 0.5f + 0.5f) * SCR_HEIGHT);
    
    float mid_x = glm::min(glm::max((screenMin.x + screenMax.x)/2, 0), SCR_WIDTH);
    float mid_y = glm::min(glm::max((screenMin.y + screenMax.y)/2, 0), SCR_HEIGHT);


    glm::vec3 middleLeft = computeRd(mid_x-(int)(difx*0.49f), mid_y);
    glm::vec3 middleRight = computeRd(mid_x+(int)(difx*0.49f), mid_y);
    glm::vec3 middleBottom = computeRd(mid_x, mid_y-(int)(dify*0.49f));
    glm::vec3 middleTop = computeRd(mid_x, mid_y+(int)(dify*0.49f));
    glm::vec3 middlePixelsVec = computeRd((screenMin.x + screenMax.x) / 2, (screenMin.y + screenMax.y) / 2);
    
    std::vector<PixelZ> spherePixels = {
        PixelZ(mid_x-(int)(difx*0.49f), mid_y, onSphDepth(middleLeft, mid_x-(int)(difx*0.49f), mid_y)),
        PixelZ(mid_x+(int)(difx*0.49f), mid_y, onSphDepth(middleRight, mid_x+(int)(difx*0.49f), mid_y)),
        PixelZ(mid_x, mid_y-(int)(dify*0.49f), onSphDepth(middleBottom, mid_x, mid_y-(int)(dify*0.49f))),
        PixelZ(mid_x, mid_y+(int)(dify*0.49f), onSphDepth(middleTop, mid_x, mid_y+(int)(dify*0.49f))),
        PixelZ((screenMin.x + screenMax.x) / 2, (screenMin.y + screenMax.y) / 2, onSphDepth(middlePixelsVec, (screenMin.x + screenMax.x) / 2, (screenMin.y + screenMax.y) / 2))
    };
    
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
        glm::vec3 rayColStart = rayStart + float(px) * dx;
        bool finishedLine = false;
        for (int py = std::max(0, screenMin.y); py < std::min(SCR_HEIGHT, screenMax.y); py++)
        {

            const glm::vec3 rd = computeRd(px, py);
            const float h = iSphere(camPos, rd, spherePos, sphere.w);

            const int index = (SCR_HEIGHT - py - 1) * SCR_WIDTH + px;
            // const bool showBbox = (px == screenMin.x || py == screenMin.y || px == screenMax.x - 1 || py == screenMax.y - 1);
            // if (showBbox) framebuffer[index] = vecToColor(glm::vec3(0));
            if ((h > 0.0f)) 
            {
                if (!finishedLine) finishedLine = true;

                const glm::vec3 hit = (rayColStart + float(py) * dy) * h;
                const float depth = (hit.z * proj[2].z + proj[3].z) / -hit.z;
                if (depth < depthBuffer[index]) 
                {
                    const glm::vec3 normal = glm::normalize( camPos + rd*h - spherePos );
                    const float lambertCos = glm::dot(normal, -glm::normalize(rd*h));
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
    // framebuffer[(SCR_HEIGHT - ((centerNdc.y * 0.5f + 0.5f) * SCR_HEIGHT) - 1) * SCR_WIDTH + ((centerNdc.x) * 0.5f + 0.5f) * SCR_WIDTH] = vecToColor(glm::vec3(0));
    return true;
}

bool drawSphere(const glm::mat4& proj, const glm::mat4& view, 
    const glm::vec3& up, const glm::vec3& front, const glm::vec3& right, const glm::vec3& camPos, 
    const int SCR_WIDTH, const int SCR_HEIGHT, const float& fov, 
    const glm::vec4& sphere,
    std::vector<uint32_t>& framebuffer, std::vector<float>& depthBuffer)
{
    glm::vec3 spherePos = glm::vec3(sphere[0], sphere[1], sphere[2]);
    glm::vec3 cameraSpaceSphere = glm::vec3(view * glm::vec4(spherePos, 1.0f));
    glm::vec3 normCamSpaceSphere = glm::normalize(cameraSpaceSphere);
    glm::vec3 camImposPos = cameraSpaceSphere - normCamSpaceSphere * sphere.w;

    bboxCorners sphereBbox = getSphereBbox(cameraSpaceSphere, camImposPos,
        normCamSpaceSphere, proj, camPos, front, up, sphere.w);
    
    float aspectRatio = (float)SCR_WIDTH / (float)SCR_HEIGHT;
    glm::ivec2 screenMin, screenMax;

    screenMin.x = ((sphereBbox.minCorner.x) * 0.5f + 0.5f) * SCR_WIDTH;
    screenMin.y = (sphereBbox.minCorner.y * 0.5f + 0.5f) * SCR_HEIGHT;
    screenMax.x = ((sphereBbox.maxCorner.x) * 0.5f + 0.5f) * SCR_WIDTH;
    screenMax.y = (sphereBbox.maxCorner.y * 0.5f + 0.5f) * SCR_HEIGHT;

    const float difx = screenMax.x - screenMin.x;
    const float dify = screenMax.y - screenMin.y;
    
    if (difx*dify <= 2) return false;  
    
    float fovRad = glm::radians(fov);
    float fovTan = tan(fovRad * 0.5f);
    float halfFovTan = fovTan * aspectRatio;

    ScreenRayCasting screenRayCasting(fov, aspectRatio, SCR_WIDTH, SCR_HEIGHT, right, up, front, view);

    // Delta per pixel
    glm::vec3 dx = screenRayCasting.dx;
    glm::vec3 dy = screenRayCasting.dy;

    // Start ray origin in world
    glm::vec3 rayStart = screenRayCasting.rayStart;

    #pragma omp parallel for collapse(2)
    for (int px = std::max(0, screenMin.x); px < std::min(screenMax.x, SCR_WIDTH); px++)
    {
        glm::vec3 rayColStart = rayStart + float(px) * dx;
        bool finishedLine = false;
        for (int py = std::max(0, screenMin.y); py < std::min(SCR_HEIGHT, screenMax.y); py++)
        {
            
            glm::vec2 p = (-glm::vec2(SCR_WIDTH, SCR_HEIGHT) + 2.0f*glm::vec2(px, py))/glm::vec2(SCR_WIDTH, SCR_HEIGHT);
            glm::vec3 rd = glm::normalize( p.x*right* halfFovTan + p.y*up * fovTan + front );
            const float h = iSphere(camPos, rd, spherePos, sphere.w);

            const int index = (SCR_HEIGHT - py - 1) * SCR_WIDTH + px;
            // const bool showBbox = (px == screenMin.x || py == screenMin.y || px == screenMax.x - 1 || py == screenMax.y - 1);
            // if (showBbox) framebuffer[index] = vecToColor(glm::vec3(0));
            if ((h > 0.0f)) 
            {
                if (!finishedLine) finishedLine = true;
                const glm::vec3 hit = (rayColStart + float(py) * dy) * h;
                
                const float depth = (hit.z * proj[2].z + proj[3].z) / -hit.z;
                if (depth < depthBuffer[index]) 
                {
                    const glm::vec3 normal = glm::normalize( camPos + rd*h - spherePos );
                    const float lambertCos = glm::dot(normal, -glm::normalize(rd*h));
                    depthBuffer[index] = depth;
                    framebuffer[index] = vecToColor(255 * lambertCos * lightColor * diffuseI);
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