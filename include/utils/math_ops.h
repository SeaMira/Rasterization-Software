#ifndef _MATH_OPS_
#define _MATH_OPS_

#include <glm/glm.hpp>

struct Interpolation
{
    inline bool nextStep(const float dt)
    {
        m_current = m_origin*(1 - t) + m_target * t;
        t += dt*m_speed;
        if (t >= 1.0f) return true;
        return false;
    } 

    inline glm::vec3 currentStep() const { return m_current; }
    inline glm::vec3 stepAlongAxis(const glm::vec3 axis, const float dt)
    {
        float old_t = t;
        t += dt; 
        return m_origin + axis*old_t;
    }
    inline float getT() const { return t; }

    Interpolation(glm::vec3 origin) : m_origin(origin), m_target(origin) {}
    Interpolation(glm::vec3 origin, glm::vec3 target) : 
        m_origin(origin), m_target(target) {}
    Interpolation(glm::vec3 origin, glm::vec3 target, float speed) :
        m_origin(origin), m_target(target), m_speed(speed) {}

    glm::vec3 m_origin, m_current, m_target;
    float t = 0.0f, m_speed = 1.0f;
};



#endif // _MATH_OPS_