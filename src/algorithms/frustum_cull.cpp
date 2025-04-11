#include "algorithms/frustum_cull.h"

#include <cmath>
#include "ux/camera.h"


bool Plane::isOnOrForwardPlane(glm::vec4& sph) const
{
    return glm::dot(normal, glm::vec3(sph)) - distance >= -sph.w;
}

bool Plane::isOnOrForwardPlaneAABB(BBox3D& bbox) const
{
    glm::vec3 negativeVertex = bbox.mMin;

    if (normal.x >= 0) negativeVertex.x = bbox.mMax.x;
        else negativeVertex.x = bbox.mMin.x;

    if (normal.y >= 0) negativeVertex.y = bbox.mMax.y;
        else negativeVertex.y = bbox.mMin.y;

    if (normal.z >= 0) negativeVertex.z = bbox.mMax.z;
        else negativeVertex.z = bbox.mMin.z;

    return glm::dot(normal, negativeVertex) - distance >= 0;
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


bool Frustum::isCylinderInside(Cylinder& cyl) const
{
    glm::vec3 a = cyl.pb - cyl.pa;
    glm::vec3 e = cyl.radius*sqrt( 1.0f - a*a/glm::dot(a,a) );
    
    BBox3D bbox3d = {glm::min( cyl.pa - e, cyl.pb - e ), glm::max( cyl.pa + e, cyl.pb + e )};

    return (leftFace.isOnOrForwardPlaneAABB(bbox3d) &&
        rightFace.isOnOrForwardPlaneAABB(bbox3d) &&
        farFace.isOnOrForwardPlaneAABB(bbox3d) &&
        nearFace.isOnOrForwardPlaneAABB(bbox3d) &&
        topFace.isOnOrForwardPlaneAABB(bbox3d) &&
        bottomFace.isOnOrForwardPlaneAABB(bbox3d));

}