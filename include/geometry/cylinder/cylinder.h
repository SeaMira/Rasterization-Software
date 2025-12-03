#ifndef CYLINDER_H
#define CYLINDER_H

#include <glm/glm.hpp>


/**
 * @struct Cylinder
 * 
 * @brief Represents a cylinder in 3D space.
 * 
 * The cylinder is defined by two points (extremes) and a radius. The points are represented as 4D vectors, where the last component is the radius of the cylinder.
 */
struct Cylinder
{
    glm::vec4 pa_r; ///< extreme A of the cylinder
    glm::vec4 pb_r; ///< extreme B of the cylinder

    /**
     * @brief Construct a new Cylinder object by defining its extremes and radius.
     * 
     * @param pa extreme A of the cylinder
     * @param pb extreme B of the cylinder
     * @param radius radius of the cylinder
     */
    Cylinder(const glm::vec3& pa, const glm::vec3& pb, float radius) : 
        pa_r(glm::vec4(pa.x, pa.y, pa.z, radius)), pb_r(glm::vec4(pb.x, pb.y, pb.z, radius)) 
        {}

    /**
     * @brief Construct a new Cylinder object with default values.
     */
    Cylinder() : pa_r(0.0f), pb_r(0.0f) {}
};

#endif // CYLINDER_H