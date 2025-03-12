#ifndef _FRUSTUM_CULLING_H_
#define _FRUSTUM_CULLING_H_
#include <glm/glm.hpp>

class Camera;

struct Plane
{
    glm::vec3 normal;
    float distance;

    Plane() = default;

    Plane(const glm::vec3& p1, const glm::vec3& norm) : 
        normal(glm::normalize(norm)), distance(glm::dot(normal, p1)) {}

    bool isOnOrForwardPlane(glm::vec4& sph) const;
};

struct Frustum
{
    Plane topFace;
    Plane bottomFace;

    Plane rightFace;
    Plane leftFace;

    Plane farFace;
    Plane nearFace;

    Frustum(Camera& camera);
    bool isSphereInside(glm::vec4& sph) const;
};


#endif