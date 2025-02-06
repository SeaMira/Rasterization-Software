#ifndef SPHERE_H
#define SPHERE_H

#include "utils/math_defines"
#include <glm/glm.hpp>  // Usamos glm para representar vectores y operaciones matemáticas

class Sphere 
{
public:
    // parameters
    glm::vec3 m_position;  // Sphere position (x, y, z)
    float m_radius;        // Sphere radius
    glm::vec3 color = glm::vec3(1.f);
    float visibility = 1.f;

    // Default constructor
    Sphere() : m_position(glm::vec3(0.0f)), m_radius(1.0f) {}

    // Parameterized constructor
    Sphere(glm::vec3 pos, float r) : m_position(pos), m_radius(r) {}

    // Setting methods
    void setPosition(const glm::vec3& pos) { m_position = pos; }
    void setRadius(float r) { m_radius = r; }

    // Getting methods
    glm::vec3 getPosition() const { return m_position; }
    float getRadius() const { return m_radius; }

    float getArea() const { return _PI * m_radius * m_radius; }
};

class SphereHolder 
{

}

#endif // SPHERE_H
