#ifndef BENCHMARKS_RESOURCES_H
#define BENCHMARKS_RESOURCES_H

#include <SDL3/SDL.h>
#include <vector>
#include <utility>
#include <filesystem>
#include <string>
#include <glm/glm.hpp>

#include "geometry/cylinder/cylinder.h"

/**
 * @enum SceneType
 * @brief Enumeration for the type of scene to load.
 * 
 * Enumeration that tells which kind of scene should be created and how spheres should be distributed.
 */
enum class SceneType {
    LOADED_SCENE,
    GRID_SCENE,
    PACKAGE_SCENE
};

#define BENCHMARKING 0

/** 
 * @brief Global variable with the scene type to load.
 */
extern SceneType currentScene;

/** 
 * @brief Global variable with the file name to load a scene from.
 */
extern std::string scene_file;

/** 
 * @brief Global variable with the file path to load a scene from.
 */
extern std::filesystem::path scene_path;

// benchmark settings
/** 
 * @brief Global variable with the grid spaces amount along the width axis on created scenes.
 */
extern int gridWidth;
/** 
 * @brief Global variable with the grid spaces amount along the height axis on created scenes.
 */
extern int gridHeight;
/** 
 * @brief Global variable with the grid spaces amount along the depth axis on created scenes.
 */
extern int gridDepth;
/** 
 * @brief Global variable with the grid width interleave for benchmark checkpoints.
 */
extern int interleaveW;
/** 
 * @brief Global variable with the grid height interleave for benchmark checkpoints.
 */
extern int interleaveH;
/** 
 * @brief Global variable with the rotation angle interleave for benchmark checkpoints on file loaded scenes.
 */
extern int interleaveAngle;
/** 
 * @brief Global variable with the depth interleave for benchmark checkpoints on file loaded scenes.
 */
extern int interleaveZ;
/** 
 * @brief Global variable with the height interleave for benchmark checkpoints on file loaded scenes.
 */
extern int interleaveY;
/** 
 * @brief Global variable with an extra radius factor for benchmark checkpoints on file loaded scenes.
 */
extern float radFactor;

/**
 * @brief Load a scene from a file utilizing a basic chemfiles loader.
 * 
 * Uses a ChemFilesLoader object to load the atoms from a file in a spheres vector.abort
 * 
 * @param path Path to the file to load the scene from.
 * @param sphere_count Number of spheres to load from the file.
 * 
 * @return vec4 vector with the spheres positions.
 */
std::vector<glm::vec4> loaded_scene(std::filesystem::path& path, int sphere_count);

/**
 * @brief Load a scene from a file utilizing a basic chemfiles loader.
 * 
 * Uses a ChemFilesLoader object to load the bonds from a file in a cylinder vector.
 * 
 * @param path Path to the file to load the scene from.
 * @param cylinder_count Number of cylinders to load from the file.
 * 
 * @return Cylinder vector with the cylinders .
 */
std::vector<Cylinder> loaded_cylinder_scene(std::filesystem::path& path, int cylinder_count);

/**
 * @brief Create a grid scene with spheres distributed in a 2D grid pattern.
 * 
 * Creates a grid scene with spheres distributed in a 2D grid pattern with as much spheres as indicated
 * in the spheres count given. Global variable used to know the grid width
 * 
 * @param gridWidth Number of spaces along the width of the grid to create.
 * @param sphere_count Number of spheres to create.
 * 
 * @return vec4 vector with the spheres positions.
 */
std::vector<glm::vec4> grid_scene(int gridWidth, int sphere_count);

/**
 * @brief Create a grid scene with cylinders distributed in a 2D grid pattern.
 * 
 * Creates a grid scene with cylinders distributed in a 2D grid pattern with as much cylinders as indicated
 * in the cylinders count given. Global variable used to know the grid width
 * 
 * @param gridWidth Number of spaces along the width of the grid to create.
 * @param cylinder_count Number of cylinders to create.
 * 
 * @return vec4 vector with the cylinders positions.
 */
std::vector<Cylinder> cylinder_grid_scene(int gridWidth, int cylinder_count);

/**
 * @brief Create a package scene with spheres distributed in a 3D grid pattern.
 * 
 * Creates a package scene with spheres distributed in a 3D grid pattern with as much spheres as indicated
 * in the spheres count given. Global variables used to know the grid width, height and depth.
 * 
 * @param gridWidth Width of the grid to create.
 * @param gridHeight Height of the grid to create.
 * @param gridDepth Depth of the grid to create.
 * @param sphere_count Number of spheres to create.
 * 
 * @return vec4 vector with the spheres positions.
 */
std::vector<glm::vec4> package_scene(int gridWidth, int gridHeight, int gridDepth, int sphere_count);

/**
 * @brief Create a package scene with cylinders distributed in a 3D grid pattern.
 * 
 * Creates a package scene with cylinders distributed in a 3D grid pattern with as much cylinders as indicated
 * in the cylinders count given. Global variables used to know the grid width, height and depth.
 * 
 * @param gridWidth Width of the grid to create.
 * @param gridHeight Height of the grid to create.
 * @param gridDepth Depth of the grid to create.
 * @param cylinder_count Number of cylinders to create.
 * 
 * @return Cylinder vector with the cylinders positions.
 */
std::vector<Cylinder> cylinder_package_scene(int gridWidth, int gridHeight, int gridDepth, int cylinder_count);

/**
 * @brief Gets the strategic checkpoints and camera targets for a benchmarck on an specific scene.
 * 
 * Creates benchmark checkpoints and targets for the camera, depending on the amount of spheres and distance from one point to another on every axis in the 2D
 * grid that distributes the spheres.
 * 
 * @param sphere_count a reference to the value that represents the amount of spheres on the scene.
 * @param gridWidth the spaces along the width of the grid.
 * @param interleaveW the distance between checkpoints on the width axis.
 * @param interleaveH the distance between checkpoints on the height axis.
 * @param interleaveZ the distance between checkpoints on the depth axis.
 * 
 * @return vector of pairs with the checkpoints and camera targets.
 */
std::vector<std::pair<glm::vec3, glm::vec3>> benchmark1_structured_grid(int& sphere_count, int gridWidth, int interleaveW, int interleaveH, int interleaveZ);

/**
 * @brief Gets the strategic checkpoints and camera targets for a benchmarck on an specific scene.
 * 
 * Creates benchmark checkpoints and targets for the camera, depending on the spheres positions 
 * and distance from the mass center of the spheres. The checkpoints rotate around the mass center at diferent heights and by a certain angle delta.
 * 
 * @param spheres a reference to the vector with the spheres positions.
 * @param cylinders a reference to the vector with the cylinders positions.
 * @param interleaveAngle the angle delta between each rotation checkpoints.
 * @param interleaveZ the distance between checkpoints on the depth axis.
 * @param interleaveY the distance between checkpoints on the height axis.
 * @param radFactor the radius factor to be used for the distance from the mass center of the spheres.
 * 
 * @return vector of pairs with the checkpoints and camera targets.
 */
std::vector<std::pair<glm::vec3, glm::vec3>> benchmark2_loaded_molecules(std::vector<glm::vec4>& spheres, std::vector<Cylinder>& cylinders, int interleaveAngle, int interleaveZ, int interleaveY, float radFactor);

/**
 * @brief Gets the strategic checkpoints and camera targets for a benchmarck on an specific scene.
 * 
 * Creates benchmark checkpoints and targets for the camera, depending on the spheres positions 
 * and distance from the mass center of the spheres. The checkpoints rotate around the mass center at diferent heights and by a certain angle delta.
 * 
 * @param spheres a reference to the vector with the spheres positions.
 * @param interleaveAngle the angle delta between each rotation checkpoints.
 * @param interleaveZ the distance between checkpoints on the depth axis.
 * @param interleaveY the distance between checkpoints on the height axis.
 * @param radFactor the radius factor to be used for the distance from the mass center of the spheres.
 * 
 * @return vector of pairs with the checkpoints and camera targets.
 */
std::vector<std::pair<glm::vec3, glm::vec3>> benchmark2_loaded_molecules(std::vector<glm::vec4>& spheres, int interleaveAngle, int interleaveZ, int interleaveY, float radFactor);

/**
 * @brief Gets the strategic checkpoints and camera targets for a benchmarck on an specific scene (3D regular spheres grid).
 * 
 * Creates benchmark checkpoints and targets for the camera, depending on the amount of spheres and distance from one point to another on every axis in the 3D grid.
 * 
 * @param sphere_count a reference to the value that represents the amount of spheres on the scene.
 * @param gridWidth the width of the grid that distributes the spheres.
 * @param gridHeight the height of the grid that distributes the spheres.
 * @param gridDepth the depth of the grid that distributes the spheres.
 * @param interleaveW the distance between checkpoints on the width axis.
 * @param interleaveH the distance between checkpoints on the height axis.
 * @param interleaveZ the distance between checkpoints on the depth axis.
 * 
 * @return vector of pairs with the checkpoints and camera targets.
 */
std::vector<std::pair<glm::vec3, glm::vec3>> benchmark3_package(int& sphere_count, int gridWidth, int gridHeight, int gridDepth, int interleaveW, int interleaveH, int interleaveZ);

/**
 * @brief Gets the vector of spheres depending on the global variables values set.
 * 
 * Gets the vector of spheres depending on the global variables values set. The global variable currentScene is used to know which kind of scene to create and 
 * the spheres amount to save.
 * 
 * @param sphere_count reference to the amount of spherez
 */
std::vector<glm::vec4> getScene(int sphere_count);

/**
 * @brief Gets the vector of cylinders depending on the global variables values set.
 * 
 * Gets the vector of cylinders depending on the global variables values set. The global variable currentScene is used to know which kind of scene to create and 
 * the cylinders amount to save.
 * 
 * @param cylinder_count reference to the amount of spherez
 */
std::vector<Cylinder> getCylinderScene(int cylinder_count);

/**
 * @brief Gets the strategic checkpoints and camera targets for a benchmarck on an specific scene.
 * 
 * Given the setted global variables, this function returns the strategic checkpoints and camera targets for a benchmarck on an specific scene depending also on the distribution of the
 * spheres in the vector.
 * 
 * @param sphere_count a reference to the value that represents the amount of spheres on the scene.
 * @param spheres reference to the vector with the spheres positions.
 * @param cylinders reference to the vector with the cylinders positions.
 * 
 * @return vector of pairs with the checkpoints and camera targets.
 */
std::vector<std::pair<glm::vec3, glm::vec3>> getCheckpoints(int& sphere_count, std::vector<glm::vec4>& spheres, std::vector<Cylinder>& cylinders);

/**
 * @brief Gets the strategic checkpoints and camera targets for a benchmarck on an specific scene.
 * 
 * Given the setted global variables, this function returns the strategic checkpoints and camera targets for a benchmarck on an specific scene depending also on the distribution of the
 * spheres in the vector.
 * 
 * @param sphere_count a reference to the value that represents the amount of spheres on the scene.
 * @param spheres reference to the vector with the spheres positions.
 * 
 * @return vector of pairs with the checkpoints and camera targets.
 */
std::vector<std::pair<glm::vec3, glm::vec3>> getCheckpoints(int& sphere_count, std::vector<glm::vec4>& spheres);

/**
 * @brief Takes a screenshot of the current renderer.
 * 
 * Takes a screenshot of the current renderer and saves it to a file with the name given.
 * 
 * @param renderer the SDL renderer to take the screenshot from.
 * @param sshot_name the name of the file to save the screenshot.
 */
void takeScreenshot(SDL_Renderer* renderer, std::string& sshot_name);

#endif // BENCHMARKS_RESOURCES_H