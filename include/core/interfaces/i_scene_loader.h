#ifndef _I_SCENE_LOADER_H_
#define _I_SCENE_LOADER_H_

#include <memory>
#include <filesystem>

// Forward declarations
class Scene;

/**
 * @interface ISceneLoader
 * 
 * @brief Interface for scene loading strategies.
 * 
 * Defines a common interface for loading scenes from different sources
 * (files, procedural generation, etc.). Follows the Open/Closed Principle (OCP)
 * by allowing new loading strategies to be added without modifying existing code.
 * 
 * @note Implementations should handle all necessary resource loading and
 *       scene construction.
 */
class ISceneLoader
{
public:
    /**
     * @brief Virtual destructor for proper cleanup in derived classes.
     */
    virtual ~ISceneLoader() = default;

    /**
     * @brief Loads a scene from a source.
     * 
     * @return Unique pointer to the loaded scene.
     * @throws std::runtime_error if loading fails.
     */
    virtual std::unique_ptr<Scene> load() = 0;

    /**
     * @brief Checks if this loader can handle the given source.
     * 
     * @param path Path to the scene source.
     * @return True if this loader can handle the source.
     */
    virtual bool canLoad(const std::filesystem::path& path) const = 0;

    /**
     * @brief Gets a descriptive name for this loader type.
     * 
     * @return Loader type name (e.g., "MMTF Loader", "Grid Generator").
     */
    virtual std::string getLoaderName() const = 0;
};

#endif // _I_SCENE_LOADER_H_
