#include "algorithms/frustum_cull.h"

#include <cmath>
#include "ux/camera.h"

bool Plane::isOnOrForwardPlane(glm::vec4& sph) const
{
    return glm::dot(normal, glm::vec3(sph)) - distance >= -sph.w;
}

Frustum::Frustum(Camera& camera)
{
    const float halfVSide = camera.getFar() * tanf(glm::radians(camera.getFov()) * .5f);
    const float halfHSide = halfVSide * (camera.SCR_WIDTH / camera.SCR_HEIGHT);
    const glm::vec3 frontMultFar = camera.getFar() * camera.getFront();

    glm::vec3 camPos = camera.getPosition();
    glm::vec3 camFront = camera.getFront();
    glm::vec3 camRight = camera.getRight();
    glm::vec3 camUp = camera.getUp();

    nearFace = Plane(camPos + camera.getNear() * camFront, camFront);
    farFace = Plane(camPos + frontMultFar, -camFront);
    rightFace = Plane(camPos,
                    glm::cross(frontMultFar - camRight * halfHSide, camUp));
    leftFace = Plane(camPos,
                    glm::cross(camUp,frontMultFar + camRight * halfHSide));
    topFace = Plane(camPos,
                    glm::cross(camRight, frontMultFar - camUp * halfVSide));
    bottomFace = Plane(camPos,
                            glm::cross(frontMultFar + camUp * halfVSide, camRight));
}

bool Frustum::isSphereInside(glm::vec4& sph) const
{
    return (leftFace.isOnOrForwardPlane(sph) &&
        rightFace.isOnOrForwardPlane(sph) &&
        farFace.isOnOrForwardPlane(sph) &&
        nearFace.isOnOrForwardPlane(sph) &&
        topFace.isOnOrForwardPlane(sph) &&
        bottomFace.isOnOrForwardPlane(sph));
}