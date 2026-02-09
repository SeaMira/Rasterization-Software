#ifndef _SCENE_BUILDER_H_
#define _SCENE_BUILDER_H_

#include <memory>
#include <vector>
#include <filesystem>
#include <glm/glm.hpp>

#include "core/scene/scene.h"
#include "core/config/scene_config.h"
#include "core/interfaces/i_scene_loader.h"

/**
 * @class SceneBuilder
 * 
 * @brief Builder class for constructing Scene objects.
 * 
 * Provides a fluent interface for building scenes from various sources
 * (files, procedural generation, etc.). Follows the Builder Pattern
 * for flexible scene construction.
 * 
 * Also follows the Dependency Inversion Principle (DIP) by depending
 * on the ISceneLoader interface rather than concrete loaders.
 */
class SceneBuilder
{
public:
    /**
     * @brief Constructs a SceneBuilder with default settings.
     */
    SceneBuilder() = default;

    /**
     * @brief Sets the scene configuration.
     * 
     * @param config Scene configuration.
     * @return Reference to this builder for chaining.
     */
    SceneBuilder& withConfig(const SceneConfig& config);

    /**
     * @brief Adds a scene loader strategy.
     * 
     * @param loader Scene loader implementation.
     * @return Reference to this builder for chaining.
     */
    SceneBuilder& withLoader(std::unique_ptr<ISceneLoader> loader);

    /**
     * @brief Adds spheres directly to the scene.
     * 
     * @param spheres Vector of spheres to add.
     * @return Reference to this builder for chaining.
     */
    SceneBuilder& withSpheres(const std::vector<geometry::Sphere>& spheres);

    /**
     * @brief Adds spheres from vec4 data.
     * 
     * @param sphereData Vector of vec4 (x, y, z, radius).
     * @return Reference to this builder for chaining.
     */
    SceneBuilder& withSpheresFromVec4(const std::vector<glm::vec4>& sphereData);

    /**
     * @brief Adds cylinders directly to the scene.
     * 
     * @param cylinders Vector of cylinders to add.
     * @return Reference to this builder for chaining.
     */
    SceneBuilder& withCylinders(const std::vector<CylinderIndex>& cylinders);

    /**
     * @brief Builds the scene.
     * 
     * @return Unique pointer to the constructed scene.
     */
    std::unique_ptr<Scene> build();

private:
    SceneConfig m_config;
    std::unique_ptr<ISceneLoader> m_loader;
    std::vector<geometry::Sphere> m_spheres;
    std::vector<glm::vec4> m_sphereData;
    std::vector<CylinderIndex> m_cylinders;
    bool m_hasDirectSpheres = false;
    bool m_hasSphereData = false;
    bool m_hasDirectCylinders = false;
};

#endif // _SCENE_BUILDER_H_
