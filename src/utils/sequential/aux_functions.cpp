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
    const glm::vec3& front, const glm::vec3& up,const float sphRadius, const float fov, const float aspectRatio)
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
    
    const glm::vec3 ndcUpRight = glm::vec3(upRightCorner) / upRight.w;
    const glm::vec3 ndcUpLeft = glm::vec3(upLeftCorner) / upLeft.w;
    const glm::vec3 ndcDownRight = glm::vec3(downRightCorner) / downRight.w;
    const glm::vec3 ndcDownLeft = glm::vec3(downLeftCorner) / downLeft.w;

    glm::vec2 minCorner = glm::min(glm::min(ndcUpRight, ndcUpLeft), glm::min(ndcDownRight, ndcDownLeft));
    glm::vec2 maxCorner = glm::max(glm::max(ndcUpRight, ndcUpLeft), glm::max(ndcDownRight, ndcDownLeft));


    return {upRightCorner, upLeftCorner, downRightCorner, downLeftCorner, minCorner, maxCorner}; 
}

uint32_t vecToColor(glm::vec3 lambertCos) 
{
    const uint8_t R = lambertCos.x;
    const uint8_t G = lambertCos.y;
    const uint8_t B = lambertCos.z;

    return (0xFF000000) | (R << 16) | (G << 8) | B;
}

void drawSphere(const glm::mat4& proj, const glm::mat4& view, 
    const glm::vec3& up, const glm::vec3& front, const glm::vec3& camPos, 
    const int SCR_WIDTH, const int SCR_HEIGHT,
    const float fov, const float aspectRatio, 
    const std::pair<glm::vec3, float>& sphere,
    std::vector<uint32_t>& framebuffer, std::vector<float>& depthBuffer)
{
    glm::vec3 cameraSpaceSphere = glm::vec3(view * glm::vec4(sphere.first, 1.0f));
    glm::vec3 normCamSpaceSphere = glm::normalize(cameraSpaceSphere);
    glm::vec3 camImposPos = cameraSpaceSphere - normCamSpaceSphere * sphere.second;

    bboxCorners sphereBbox = getSphereBbox(cameraSpaceSphere, camImposPos,
        normCamSpaceSphere, proj, camPos, front, up, sphere.second, fov, aspectRatio);
    
        glm::ivec2 screenMin, screenMax;

    screenMin.x = ((sphereBbox.minCorner.x/aspectRatio) * 0.5f + 0.5f) * SCR_WIDTH;
    screenMin.y = (sphereBbox.minCorner.y * 0.5f + 0.5f) * SCR_HEIGHT;
    screenMax.x = ((sphereBbox.maxCorner.x/aspectRatio) * 0.5f + 0.5f) * SCR_WIDTH;
    screenMax.y = (sphereBbox.maxCorner.y * 0.5f + 0.5f) * SCR_HEIGHT;
    // std::cout << screenMin.x << ", " << screenMin.y << std::endl;
    // std::cout << screenMax.x << ", " << screenMax.y << std::endl;   

    if (glm::dot(sphere.first - camPos, front) > 0.5f && !((screenMin.x < 0 && screenMax.x > SCR_WIDTH) || (screenMin.y < 0 && screenMax.y > SCR_HEIGHT))) 
    {
        const float difx = screenMax.x - screenMin.x;
        const float dify = screenMax.y - screenMin.y;
        
        #pragma omp parallel for collapse(2)
        for (int px = std::max(0, screenMin.x); px < std::min(screenMax.x, SCR_WIDTH); px++)
        {
            for (int py = std::max(0, screenMin.y); py < std::min(SCR_HEIGHT, screenMax.y); py++)
            {

                const float u = (float)(px - screenMin.x)/ difx;
                const float v = (float)(py - screenMin.y)/ dify;
                const glm::vec3 A = glm::mix(sphereBbox.upLeftCorner, sphereBbox.upRightCorner, u);
                const glm::vec3 B = glm::mix(sphereBbox.downLeftCorner, sphereBbox.downRightCorner, u);
                const glm::vec3 viewImpPos = glm::mix(A, B, v);
                const float h = iSphere(camPos, viewImpPos, cameraSpaceSphere, sphere.second);

                const bool showBbox = (px == screenMin.x || py == screenMin.y || px == screenMax.x - 1 || py == screenMax.y - 1);
                if ((h > 0.0f)) 
                {
                    const int index = (SCR_HEIGHT - py - 1) * SCR_WIDTH + px;
                    const glm::vec3 hit = viewImpPos * h;
                    const float depth = hit.z < 0.0f ? (hit.z * proj[2].z + proj[3].z) / -hit.z : FLT_MAX;
                    if (depth < depthBuffer[index]) 
                    {
                        const glm::vec3 normal = glm::normalize( hit - cameraSpaceSphere );
                        const float lambertCos = glm::dot(normal, -glm::normalize(hit));
                        depthBuffer[index] = depth;
                        framebuffer[index] = vecToColor(255 * lambertCos * lightColor * diffuseI);
                    }
                }
            }
        }
    }
}