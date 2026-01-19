#ifndef _FILE_SCENE_LOADER_H_
#define _FILE_SCENE_LOADER_H_

#include <memory>
#include <filesystem>

#include "core/interfaces/i_scene_loader.h"
#include "core/config/scene_config.h"

/**
 * @class FileSceneLoader
 * 
 * @brief Loads scenes from molecular data files.
 * 
 * Implements ISceneLoader for loading scenes from supported
 * molecular file formats (MMTF, PDB, etc.). Uses the chemfiles
 * library for parsing.
 * 
 * Follows the Single Responsibility Principle (SRP) by focusing
 * solely on file-based scene loading.
 * 
 * Also demonstrates the Strategy Pattern as one of multiple
 * possible scene loading strategies.
 */
class FileSceneLoader : public ISceneLoader
{
public:
    /**
     * @brief Constructs a file scene loader.
     * 
     * @param config Scene configuration with file path.
     */
    explicit FileSceneLoader(const SceneConfig& config);

    /**
     * @brief Default destructor.
     */
    ~FileSceneLoader() override = default;

    /**
     * @brief Loads the scene from the configured file.
     * 
     * @return Unique pointer to the loaded scene.
     * @throws std::runtime_error if loading fails.
     */
    std::unique_ptr<Scene> load() override;

    /**
     * @brief Checks if this loader can handle the given path.
     * 
     * Supports: .mmtf, .pdb, .mol2, .xyz formats.
     * 
     * @param path Path to check.
     * @return True if format is supported.
     */
    bool canLoad(const std::filesystem::path& path) const override;

    /**
     * @brief Gets the loader type name.
     * 
     * @return "File Scene Loader".
     */
    std::string getLoaderName() const override { return "File Scene Loader"; }

private:
    SceneConfig m_config;   ///< Scene configuration
};

/**
 * @class GridSceneLoader
 * 
 * @brief Generates procedural grid scenes.
 * 
 * Implements ISceneLoader for creating procedurally generated
 * grid-based scenes for testing and benchmarking.
 * 
 * Follows the Open/Closed Principle (OCP) - new scene generation
 * strategies can be added without modifying existing loaders.
 */
class GridSceneLoader : public ISceneLoader
{
public:
    /**
     * @brief Constructs a grid scene loader.
     * 
     * @param config Scene configuration with grid parameters.
     */
    explicit GridSceneLoader(const SceneConfig& config);

    /**
     * @brief Default destructor.
     */
    ~GridSceneLoader() override = default;

    /**
     * @brief Generates the grid scene.
     * 
     * @return Unique pointer to the generated scene.
     */
    std::unique_ptr<Scene> load() override;

    /**
     * @brief Grid loader can always load (doesn't need a file).
     * 
     * @param path Ignored for grid generation.
     * @return Always true.
     */
    bool canLoad(const std::filesystem::path& path) const override { return true; }

    /**
     * @brief Gets the loader type name.
     * 
     * @return "Grid Scene Loader".
     */
    std::string getLoaderName() const override { return "Grid Scene Loader"; }

private:
    /**
     * @brief Generates spheres in a grid pattern.
     * 
     * @param scene Scene to populate.
     */
    void generateSpheres(Scene& scene) const;

    /**
     * @brief Generates cylinders connecting adjacent spheres.
     * 
     * @param scene Scene to populate.
     */
    void generateCylinders(Scene& scene) const;

private:
    SceneConfig m_config;   ///< Scene configuration
};

/**
 * @class SceneLoaderFactory
 * 
 * @brief Factory for creating appropriate scene loaders.
 * 
 * Implements the Factory Pattern to create the correct
 * scene loader based on configuration.
 * 
 * Follows the Dependency Inversion Principle (DIP) by
 * allowing client code to work with the ISceneLoader
 * interface rather than concrete implementations.
 */
class SceneLoaderFactory
{
public:
    /**
     * @brief Creates the appropriate loader for the given config.
     * 
     * @param config Scene configuration.
     * @return Unique pointer to the appropriate loader.
     * @throws std::invalid_argument if no suitable loader exists.
     */
    static std::unique_ptr<ISceneLoader> create(const SceneConfig& config);
};

#endif // _FILE_SCENE_LOADER_H_
