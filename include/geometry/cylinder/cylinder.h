#ifndef CYLINDER_H
#define CYLINDER_H

#include <glm/glm.hpp>

struct Cylinder
{
    glm::vec3 pa; // extreme A
    glm::vec3 pb; // extreme B
    float radius;

    Cylinder(const glm::vec3& pa, const glm::vec3& pb, float radius) : pa(pa), pb(pb), radius(radius) {}
    Cylinder() : pa(0.0f), pb(0.0f), radius(0.0f) {}
};

#endif // CYLINDER_H