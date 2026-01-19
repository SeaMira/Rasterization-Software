#include "core/scene/scene_builder.h"

SceneBuilder& SceneBuilder::withConfig(const SceneConfig& config)
{
    m_config = config;
    return *this;
}

SceneBuilder& SceneBuilder::withLoader(std::unique_ptr<ISceneLoader> loader)
{
    m_loader = std::move(loader);
    return *this;
}

SceneBuilder& SceneBuilder::withSpheres(const std::vector<geometry::Sphere>& spheres)
{
    m_spheres = spheres;
    m_hasDirectSpheres = true;
    return *this;
}

SceneBuilder& SceneBuilder::withSpheresFromVec4(const std::vector<glm::vec4>& sphereData)
{
    m_sphereData = sphereData;
    m_hasSphereData = true;
    return *this;
}

SceneBuilder& SceneBuilder::withCylinders(const std::vector<Cylinder>& cylinders)
{
    m_cylinders = cylinders;
    m_hasDirectCylinders = true;
    return *this;
}

std::unique_ptr<Scene> SceneBuilder::build()
{
    // If we have a loader, use it
    if (m_loader)
    {
        auto scene = m_loader->load();
        return scene;
    }

    // Otherwise, build manually from provided data
    auto scene = std::make_unique<Scene>(m_config);

    if (m_hasDirectSpheres)
    {
        scene->addSpheresFromClass(m_spheres);
    }

    if (m_hasSphereData)
    {
        scene->addSpheres(m_sphereData);
    }

    if (m_hasDirectCylinders)
    {
        scene->addCylinders(m_cylinders);
    }

    return scene;
}
