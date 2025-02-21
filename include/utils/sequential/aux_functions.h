#ifndef _AUX_SEQUENTIAL_H_
#define _AUX_SEQUENTIAL_H_

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <cmath>
#include <omp.h>


const glm::vec3 lightColor(0.01f, 1.0f, 0.05f);
const float diffuseI = 0.9f;

struct bboxCorners
{
    // view space
    glm::vec3 upRightCorner;
    glm::vec3 upLeftCorner;
    glm::vec3 downRightCorner;
    glm::vec3 downLeftCorner;

    // screen space
    glm::vec2 minCorner;
    glm::vec2 maxCorner;
};

float iSphere(glm::vec3 ro, glm::vec3 rd, glm::vec3 sph, float radius );


bboxCorners getSphereBbox(const glm::vec3& cameraSpaceSphere, const glm::vec3& camImposPos, 
    const glm::vec3& normCamSpaceSphere, const glm::mat4& proj, const glm::vec3& camPos, 
    const glm::vec3& front, const glm::vec3& up,const float sphRadius, const float fov, const float aspectRatio);



uint32_t vecToColor(glm::vec3 lambertCos);

void drawSphere(const glm::mat4& proj, const glm::mat4& view, 
    const glm::vec3& up, const glm::vec3& front, const glm::vec3& camPos, 
    const int SCR_WIDTH, const int SCR_HEIGHT,
    const float fov, const float aspectRatio, 
    const glm::vec4& sphere,
    std::vector<uint32_t>& framebuffer, std::vector<float>& depthBuffer);

#endif // _AUX_SEQUENTIAL_H_