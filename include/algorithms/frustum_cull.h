#ifndef _FRUSTUM_CULLING_H_
#define _FRUSTUM_CULLING_H_
#include <glm/glm.hpp>

class Camera;

/**
 * @struct Plane
 * @brief Represents a plane on 3D space.
 *
 * The `Plane` class represents a plane on 3D space through a normal and the shortest distance to it from the origin.
 * 
 * It also provides a method to check if a sphere is on or in front of the plane with a signed distance.
 */
struct Plane
{
    glm::vec3 normal; ///< normal vector of the plane
    float distance; ///< closest distance from the origin to the plane.

    Plane() = default;

    /**
     * @brief Construct a new Plane object with its normal and shortest distance to it.
     * 
     * @param p1 A point on the plane
     * @param norm The normal vector to the plane (can be with any length)
     * 
     * Takes a point on the plane and its normal vector to construct the plane by normalizing
     * the normal and the shortest distance to it with a dot product between the both of them.
     */
    Plane(const glm::vec3& p1, const glm::vec3& norm) : 
        normal(glm::normalize(norm)), distance(glm::dot(normal, p1)) {}

    /**
     * @brief Tells if a sphere is in front or on a the plane or not .
     * 
     * Uses the signed distance from a point (sphere center) to the plane  minus the radius
     * to determine if the sphere is on or in front of the plane.
     * 
     * @param sph A sphere represented by a vec4 with the center and the radius
     * 
     * @return true if the sphere is on or in front of the plane, false otherwise.
     */
    bool isOnOrForwardPlane(glm::vec4& sph) const;
};

/**
 * @struct Frustum
 * @brief Represents a camera frustum on 3D.
 *
 * The `Frustum` class represents a camera frustum on 3D space. Its the vision volume of the camera that contains its visible objects.
 * A frustum is a portion of of a pyramid with the top cut off, base parallel to the top, and can be represented with 6 planes.
 * 
 * It also provides a method to check if a sphere is inside of the frustum.
 */
struct Frustum
{
    Plane topFace;/**< Plane representing the top part of the frustum (normal facing downwards), depends on the FOV of the camera */
    Plane bottomFace; /**< Plane representing the bottom part of the frustum (normal facing upwards), depends on the FOV of the camera */

    Plane rightFace; /**< Plane representing the right part of the frustum (normal facing leftwards), depends on the FOV of the camera */
    Plane leftFace; /**< Plane representing the left part of the frustum (normal facing rightwards), depends on the FOV of the camera */

    Plane farFace; /**< Plane representing the farthest part of the frustum (normal facing to the camera), depends on the far distance of the camera */
    Plane nearFace; /**< Plane representing the nearest part of the frustum (normal exiting the camera), depends on the near distance of the camera */

    /**
     * @brief Construct a new Frustum object with a camera.
     * 
     * @param camera The camera to construct the frustum from.
     * 
     * Constructs a frustum from a camera by using its position, front, right, up vectors and the FOV, near and far distances.
     */
    Frustum(Camera& camera);

    /**
     * @brief Tells if a sphere is inside the frustum or not.
     * 
     * Uses the `isOnOrForwardPlane` method from the `Plane` structure to check if the sphere is inside the frustum.
     * If the sphere is on or in front of all the planes, it is considered inside the frustum.
     * 
     * @param sph A sphere represented by a vec4 with the center and the radius
     * 
     * @return true if the sphere is inside the frustum, false otherwise.
     */
    bool isSphereInside(glm::vec4& sph) const;
};


#endif