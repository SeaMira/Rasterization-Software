#ifndef _MATH_OPS_
#define _MATH_OPS_

#include <glm/glm.hpp>

/**
 * @struct Interpolation
 * 
 * @brief Interpolation class for linear interpolation between two points or along an axis.
 */
struct Interpolation
{
    /**
     * @brief given a delta time, calculate the next step in the interpolation.
     * 
     * Following an interpolation equation between two points, the next step is calculated
     * by incrementing the current point by a fraction of the distance (dt times speed) between the origin and target.
     * 
     * @param dt delta time
     * @return bool true if the interpolation has reached the target, false otherwise.
     */
    inline bool nextStep(const float dt)
    {
        m_current = m_origin*(1 - t) + m_target * t;
        t += dt*m_speed;
        if (t >= 1.0f) return true;
        return false;
    } 

    /**
     * @brief get the current step in the interpolation.
     * 
     * Gets the current position of the interpolation.
     * 
     * @return current step in the interpolation.
     */
    inline glm::vec3 currentStep() const { return m_current; }

    /**
     * @brief gives a step along an axis
     * 
     * Given an axis and a delta time, the next step is calculated by incrementing the current point by a fraction of the distance 
     * (dt times speed) along the axis.
     * 
     * @return the old step along the axis.
     */
    inline glm::vec3 stepAlongAxis(const glm::vec3 axis, const float dt)
    {
        float old_t = t;
        t += dt; 
        return m_origin + axis*old_t;
    }

    /**
     * @brief get the current t value.
     */
    inline float getT() const { return t; }

    /**
     * Constructor for the Interpolation class when the interpolation is standing in a point.
     * 
     * @param origin the starting point of the interpolation.
     * 
     */
    Interpolation(glm::vec3 origin) : m_origin(origin), m_target(origin) {}

    /**
     * Constructor for the Interpolation class when the interpolation is moving between two points.
     * 
     * @param origin the starting point of the interpolation.
     * @param target the ending point of the interpolation.
     */
    Interpolation(glm::vec3 origin, glm::vec3 target) : 
        m_origin(origin), m_target(target) {}

    /**
     * Constructor for the Interpolation class when the interpolation is moving between two points with a given speed.
     * 
     * @param origin the starting point of the interpolation.
     * @param target the ending point of the interpolation.
     * @param speed the speed of the interpolation.
     */
    Interpolation(glm::vec3 origin, glm::vec3 target, float speed) :
        m_origin(origin), m_target(target), m_speed(speed) {}

    glm::vec3 m_origin; ///< the starting point of the interpolation.
    glm::vec3 m_current; ///< the current point of the interpolation.
    glm::vec3 m_target; ///< the ending point of the interpolation.
    float t = 0.0f; ///< the current t value of the interpolation.
    float m_speed = 1.0f; ///< the speed of the interpolation.
};



#endif // _MATH_OPS_