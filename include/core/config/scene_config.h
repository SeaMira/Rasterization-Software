#ifndef _SCENE_CONFIG_H_
#define _SCENE_CONFIG_H_

#include <string>
#include <filesystem>

/**
 * @enum SceneSourceType
 * 
 * @brief Enumeration for scene data source types.
 * 
 * Defines the different ways a scene can be constructed:
 * from a molecular file, procedurally generated grid, or
 * a packed/custom scene.
 */
enum class SceneSourceType
{
    FILE_LOADED,    ///< Scene loaded from a molecular data file
    GRID_GENERATED, ///< Procedurally generated grid scene
    PACKED_SCENE    ///< Custom packed scene configuration
};

/**
 * @class SceneConfig
 * 
 * @brief Configuration class for scene parameters.
 * 
 * Encapsulates all scene configuration parameters, replacing
 * scattered global variables. Follows the Single Responsibility
 * Principle (SRP) for scene configuration management.
 * 
 * Uses the Builder pattern for flexible configuration.
 */
class SceneConfig
{
public:
    /**
     * @class Builder
     * 
     * @brief Builder for constructing SceneConfig instances.
     */
    class Builder; // Forward declaration; defined after SceneConfig is complete

    // Getters

    /**
     * @brief Gets the scene source type.
     * @return Scene source type enumeration value.
     */
    SceneSourceType getSourceType() const { return m_sourceType; }

    /**
     * @brief Gets the scene file path.
     * @return Path to the scene file.
     */
    const std::filesystem::path& getFilePath() const { return m_filePath; }

    /**
     * @brief Gets the scene file name.
     * @return Scene file name without path.
     */
    const std::string& getFileName() const { return m_fileName; }

    /**
     * @brief Gets the expected sphere count.
     * @return Maximum sphere count.
     */
    int getSphereCount() const { return m_sphereCount; }

    /**
     * @brief Gets the expected cylinder count.
     * @return Maximum cylinder count.
     */
    int getCylinderCount() const { return m_cylinderCount; }

    /**
     * @brief Gets the grid width.
     * @return Grid width for procedural generation.
     */
    int getGridWidth() const { return m_gridWidth; }

    /**
     * @brief Gets the grid height.
     * @return Grid height for procedural generation.
     */
    int getGridHeight() const { return m_gridHeight; }

    /**
     * @brief Gets the grid depth.
     * @return Grid depth for procedural generation.
     */
    int getGridDepth() const { return m_gridDepth; }

    /**
     * @brief Gets the sphere radius.
     * @return Sphere radius for procedural generation.
     */
    float getSphereRadius() const { return m_sphereRadius; }

    /**
     * @brief Gets the cylinder radius.
     * @return Cylinder radius.
     */
    float getCylinderRadius() const { return m_cylinderRadius; }

    /**
     * @brief Gets the entity separation.
     * @return Separation distance between entities.
     */
    float getSeparation() const { return m_separation; }

private:
    SceneSourceType m_sourceType = SceneSourceType::FILE_LOADED;
    std::filesystem::path m_filePath;
    std::string m_fileName;
    
    int m_sphereCount = 0;
    int m_cylinderCount = 0;
    
    int m_gridWidth = 10;
    int m_gridHeight = 10;
    int m_gridDepth = 10;
    
    float m_sphereRadius = 1.0f;
    float m_cylinderRadius = 0.3f;
    float m_separation = 2.0f;
};

/**
 * @class SceneConfig::Builder
 * 
 * @brief Builder implementation for constructing SceneConfig instances.
 */
class SceneConfig::Builder
{
public:
    /**
     * @brief Sets the scene source type.
     * 
     * @param type Type of scene source.
     * @return Reference to this builder for chaining.
     */
    Builder& sourceType(SceneSourceType type) { m_config.m_sourceType = type; return *this; }

    /**
     * @brief Sets the scene file path (for FILE_LOADED type).
     * 
     * @param path Path to the scene file.
     * @return Reference to this builder for chaining.
     */
    Builder& filePath(const std::filesystem::path& path)
    {
        m_config.m_filePath = path;
        m_config.m_fileName = path.filename().string();
        return *this;
    }

    /**
     * @brief Sets the expected sphere count.
     * 
     * @param count Maximum number of spheres.
     * @return Reference to this builder for chaining.
     */
    Builder& sphereCount(int count) { m_config.m_sphereCount = count; return *this; }

    /**
     * @brief Sets the expected cylinder count.
     * 
     * @param count Maximum number of cylinders.
     * @return Reference to this builder for chaining.
     */
    Builder& cylinderCount(int count) { m_config.m_cylinderCount = count; return *this; }

    /**
     * @brief Sets the grid dimensions for procedural generation.
     * 
     * @param width Grid width.
     * @param height Grid height.
     * @param depth Grid depth.
     * @return Reference to this builder for chaining.
     */
    Builder& gridDimensions(int width, int height, int depth)
    {
        m_config.m_gridWidth = width;
        m_config.m_gridHeight = height;
        m_config.m_gridDepth = depth;
        return *this;
    }

    /**
     * @brief Sets the sphere radius for procedural generation.
     * 
     * @param radius Sphere radius.
     * @return Reference to this builder for chaining.
     */
    Builder& sphereRadius(float radius) { m_config.m_sphereRadius = radius; return *this; }

    /**
     * @brief Sets the cylinder radius.
     * 
     * @param radius Cylinder radius.
     * @return Reference to this builder for chaining.
     */
    Builder& cylinderRadius(float radius) { m_config.m_cylinderRadius = radius; return *this; }

    /**
     * @brief Sets the separation between entities.
     * 
     * @param separation Entity separation distance.
     * @return Reference to this builder for chaining.
     */
    Builder& separation(float separation) { m_config.m_separation = separation; return *this; }

    /**
     * @brief Builds the SceneConfig object.
     * 
     * @return Configured SceneConfig instance.
     */
    SceneConfig buildConfig() const { return m_config; }

private:
    SceneConfig m_config;
};

#endif // _SCENE_CONFIG_H_
