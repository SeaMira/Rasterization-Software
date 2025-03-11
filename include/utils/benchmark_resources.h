#ifndef BENCHMARKS_RESOURCES_H
#define BENCHMARKS_RESOURCES_H

#include <SDL3/SDL.h>
#include <vector>
#include <utility>
#include <string>
#include <glm/glm.hpp>

// benchmark settings
extern int spheresGridWidth;
extern int interleaveW;
extern int interleaveH;

extern int interleaveAngle;
extern int interleaveZ;
extern int interleaveY;

std::vector<std::pair<glm::vec3, glm::vec3>> benchmark1_structured_grid(int& sphere_count, int spheresGridWidth, int interleaveW, int interleaveH, int interleaveZ);
std::vector<std::pair<glm::vec3, glm::vec3>> benchmark2_loaded_molecules(std::vector<glm::vec4>& spheres, int interleaveAngle, int interleaveZ, int interleaveY);
void takeScreenshot(SDL_Renderer* renderer, std::string& sshot_name);

#endif // BENCHMARKS_RESOURCES_H