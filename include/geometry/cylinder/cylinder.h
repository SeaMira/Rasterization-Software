#ifndef CYLINDER_H
#define CYLINDER_H

#include <glm/glm.hpp>

struct Cylinder
{
    glm::vec4 pa_r; // extreme A
    glm::vec4 pb_r; // extreme B

    Cylinder(const glm::vec3& pa, const glm::vec3& pb, float radius) : pa_r(glm::vec4(pa, radius)), pb_r(glm::vec4(pb, radius)) {}
    Cylinder() : pa_r(0.0f), pb_r(0.0f) {}
};

#endif // CYLINDER_H