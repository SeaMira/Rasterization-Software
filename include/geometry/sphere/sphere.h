#ifndef SPHERE_H
#define SPHERE_H

#include "utils/math_defines"
#include <glm/glm.hpp>  // Usamos glm para representar vectores y operaciones matemáticas


/**
 * * @class Sphere
 * 
 * @brief Represents a sphere in 3D space.
 * 
 * The sphere is defined by its position (center) and radius. The position is represented as a 3D vector (x, y, z).
 * The radius is a float value. The class provides methods to set and get the position and radius of the sphere.
 * 
 * It may also contain color and visibility attributes.
 */
class Sphere 
{
public:
    // parameters
    glm::vec3 m_position;  ///< Sphere position (x, y, z)
    float m_radius;        ///< Sphere radius
    glm::vec3 color = glm::vec3(1.f); ///< Sphere color (r, g, b)
    float visibility = 1.f; ///< Sphere visibility (1.0 = visible, 0.0 = not visible)

    /**
     * @brief Construct a new Sphere object with default values: position (0, 0, 0) and radius 1.0.
     */
    Sphere() : m_position(glm::vec3(0.0f)), m_radius(1.0f) {}

    /**
     * @brief Construct a new Sphere object with given position and radius.
     * 
     * @param pos Position of the sphere (x, y, z).
     * @param r Radius of the sphere.
     */
    Sphere(glm::vec3 pos, float r) : m_position(pos), m_radius(r) {}

    /**
     * @brief Sets a position to the sphere.
     * 
     * @param pos Position of the sphere (x, y, z).
     */
    void setPosition(const glm::vec3& pos) { m_position = pos; }

    /**
     * @brief Sets the radius of the sphere.
     * 
     * @param r Radius of the sphere.
     */
    void setRadius(float r) { m_radius = r; }

    /**
     * @brief Gets the position of the sphere.
     * 
     * @return Position of the sphere (x, y, z).
     */
    glm::vec3 getPosition() const { return m_position; }

    /**
     * @brief Gets the radius of the sphere.
     * 
     * @return Radius of the sphere.
     */
    float getRadius() const { return m_radius; }

    /**
     * @brief Gets the area of the sphere.
     * 
     * @return Area of the sphere.
     */
    float getArea() const { return 4.0f * _PI * m_radius * m_radius; }

    /**
     * @brief Gets the volume of the sphere.
     * 
     * @return Volume of the sphere.
     */
    float getVolume() const { return (4.0f / 3.0f) * _PI * m_radius * m_radius * m_radius; }
};

#endif // SPHERE_H
