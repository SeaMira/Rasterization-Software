#ifndef BENCHMARKS_RESOURCES_H
#define BENCHMARKS_RESOURCES_H

#include <SDL3/SDL.h>
#include <vector>
#include <utility>
#include <filesystem>
#include <string>
#include <glm/glm.hpp>

enum class SceneType {
    LOADED_SCENE,
    GRID_SCENE
};

extern SceneType currentScene;
extern std::filesystem::path scene_path;

// benchmark settings
extern int spheresGridWidth;
extern int interleaveW;
extern int interleaveH;

extern int interleaveAngle;
extern int interleaveZ;
extern int interleaveY;

std::vector<glm::vec4> loaded_scene(std::filesystem::path& path, int sphere_count);
std::vector<glm::vec4> grid_scene(int spheresGridWidth, int sphere_count);
std::vector<std::pair<glm::vec3, glm::vec3>> benchmark1_structured_grid(int& sphere_count, int spheresGridWidth, int interleaveW, int interleaveH, int interleaveZ);
std::vector<std::pair<glm::vec3, glm::vec3>> benchmark2_loaded_molecules(std::vector<glm::vec4>& spheres, int interleaveAngle, int interleaveZ, int interleaveY);

std::vector<glm::vec4> getScene(int sphere_count);
std::vector<std::pair<glm::vec3, glm::vec3>> getCheckpoints(int& sphere_count, std::vector<glm::vec4>& spheres);
void takeScreenshot(SDL_Renderer* renderer, std::string& sshot_name);

#endif // BENCHMARKS_RESOURCES_H