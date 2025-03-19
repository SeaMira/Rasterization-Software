#ifndef _AUX_SEQUENTIAL_H_
#define _AUX_SEQUENTIAL_H_

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <cmath>
#include <omp.h>

#include "algorithms/occlusion_cull.h"

const glm::vec3 lightColor(0.01f, 1.0f, 0.05f);
const float diffuseI = 0.9f;

struct bboxCorners
{
    // view space
    glm::vec3 upRightCorner;
    float upRightCornerndcZ;
    glm::vec3 upLeftCorner;
    float upLeftCornerndcZ;
    glm::vec3 downRightCorner;
    float downRightCornerndcZ;
    glm::vec3 downLeftCorner;
    float downLeftCornerndcZ;

    // screen space
    glm::vec2 minCorner;
    glm::vec2 maxCorner;


};

float iSphere(glm::vec3 ro, glm::vec3 rd, glm::vec3 sph, float radius );


bboxCorners getSphereBbox(const glm::vec3& cameraSpaceSphere, const glm::vec3& camImposPos, 
    const glm::vec3& normCamSpaceSphere, const glm::mat4& proj, const glm::vec3& camPos, 
    const glm::vec3& front, const glm::vec3& up,const float sphRadius, HierarchicalZBuffer& hizPyramid);



uint32_t vecToColor(glm::vec3 lambertCos);

bool drawSphere(const glm::mat4& proj, const glm::mat4& view, 
    const glm::vec3& up, const glm::vec3& front, const glm::vec3& camPos, 
    const int SCR_WIDTH, const int SCR_HEIGHT, 
    const glm::vec4& sphere,
    std::vector<uint32_t>& framebuffer, std::vector<float>& depthBuffer, HierarchicalZBuffer& hizPyramid);


bool drawBillboard(const glm::mat4& proj, const glm::mat4& view, 
    const glm::vec3& up, const glm::vec3& front, const glm::vec3& camPos, 
    const int SCR_WIDTH, const int SCR_HEIGHT, 
    const glm::vec4& sphere,
    std::vector<uint32_t>& framebuffer, std::vector<float>& depthBuffer, HierarchicalZBuffer& hizPyramid);

#endif // _AUX_SEQUENTIAL_H_