#include "core/scene/scene.h"
#include <algorithm>
#include <limits>

Scene::Scene(const SceneConfig& config)
    : m_config(config)
{
    // Pre-allocate memory if counts are known
    if (config.getSphereCount() > 0)
    {
        m_spheres.reserve(static_cast<size_t>(config.getSphereCount()));
    }
    if (config.getCylinderCount() > 0)
    {
        m_cylinders.reserve(static_cast<size_t>(config.getCylinderCount()));
    }
}

void Scene::addSphere(const glm::vec4& sphere)
{
    m_spheres.push_back(sphere);
}

void Scene::addSpheres(const std::vector<glm::vec4>& spheres)
{
    m_spheres.insert(m_spheres.end(), spheres.begin(), spheres.end());
}

void Scene::addSpheresFromClass(const std::vector<geometry::Sphere>& spheres)
{
    m_spheres.reserve(m_spheres.size() + spheres.size());
    for (const auto& sphere : spheres)
    {
        m_spheres.push_back(glm::vec4(sphere.m_position, sphere.m_radius));
    }
}

void Scene::addCylinder(const CylinderIndex& cylinder)
{
    m_cylinders.push_back(cylinder);
}

void Scene::addCylinders(const std::vector<CylinderIndex>& cylinders)
{
    m_cylinders.insert(m_cylinders.end(), cylinders.begin(), cylinders.end());
}

void Scene::clear()
{
    m_spheres.clear();
    m_cylinders.clear();
}

void Scene::computeBoundingBox(glm::vec3& outMin, glm::vec3& outMax) const
{
    constexpr float maxFloat = std::numeric_limits<float>::max();
    constexpr float minFloat = std::numeric_limits<float>::lowest();
    
    outMin = glm::vec3(maxFloat);
    outMax = glm::vec3(minFloat);

    // Process spheres
    for (const auto& sphere : m_spheres)
    {
        glm::vec3 sphereMin = glm::vec3(sphere) - glm::vec3(sphere.w);
        glm::vec3 sphereMax = glm::vec3(sphere) + glm::vec3(sphere.w);
        
        outMin = glm::min(outMin, sphereMin);
        outMax = glm::max(outMax, sphereMax);
    }

    // Process cylinders (resolve positions from sphere buffer)
    for (const auto& cylinder : m_cylinders)
    {
        if (cylinder.sphereIndexA < m_spheres.size() && cylinder.sphereIndexB < m_spheres.size())
        {
            glm::vec3 pa(m_spheres[cylinder.sphereIndexA]);
            glm::vec3 pb(m_spheres[cylinder.sphereIndexB]);
            float radius = cylinder.radius;

            glm::vec3 cylMin = glm::min(pa, pb) - glm::vec3(radius);
            glm::vec3 cylMax = glm::max(pa, pb) + glm::vec3(radius);

            outMin = glm::min(outMin, cylMin);
            outMax = glm::max(outMax, cylMax);
        }
    }

    // Handle empty scene
    if (isEmpty())
    {
        outMin = glm::vec3(0.0f);
        outMax = glm::vec3(0.0f);
    }
}

glm::vec3 Scene::computeCenter() const
{
    glm::vec3 min, max;
    computeBoundingBox(min, max);
    return (min + max) * 0.5f;
}
