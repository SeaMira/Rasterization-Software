#include <iostream>
#include "utils/benchmark_resources.h"
#include "molecule_loader/basic_loader.h"

// LOADED_SCENE or GRID_SCENE
SceneType currentScene = SceneType::PACKAGE_SCENE;
std::filesystem::path scene_path = "assets/molecules/1AGA.mmtf";

int spheresGridWidth = 10;
int spheresGridHeight = 10;
int spheresGridDepth = 10;
int interleaveW = 5;
int interleaveH = 5;

int interleaveAngle = 10;
int interleaveZ = 10;
int interleaveY = 10;


std::vector<glm::vec4> loaded_scene(std::filesystem::path& path, int sphere_count)
{
    ChemFilesLoader loader(path);
    std::vector<glm::vec4> positions = loader.getSphereInfo();
    std::vector<glm::vec4> spheres(positions.begin(), positions.begin() + std::min(positions.size(), static_cast<size_t>(sphere_count)));
    return spheres;
}

std::vector<glm::vec4> grid_scene(int spheresGridWidth, int sphere_count)
{
    std::vector<glm::vec4> spheres;
    for (int i = 0; i < sphere_count; i++)
    {
        spheres.push_back({(float)(i%spheresGridWidth)*2.0f, (float)(i/spheresGridWidth) * 2.0f, (float)(i%spheresGridWidth)*2.0f, 1.0f});
    }
    return spheres;
}

std::vector<glm::vec4> package_scene(int spheresGridWidth, int spheresGridHeight, int spheresGridDepth, int sphere_count)
{
    std::vector<glm::vec4> spheres;
    int count = 0;
    for (int i = 0; i < spheresGridWidth; i++)
    {
        for (int j = 0; j < spheresGridHeight; j++)
        {
            for (int k = 0; k < spheresGridDepth; k++)
            {
                if (count >= sphere_count) break;
                spheres.push_back({(float)i*2.0f, (float)j*2.0f, (float)k*2.0f, 1.0f});
                count++;
            }
        }
    }
    return spheres;
}

std::vector<std::pair<glm::vec3, glm::vec3>> benchmark1_structured_grid(int& sphere_count, int spheresGridWidth, int interleaveW, int interleaveH, int interleaveZ)
{
    std::vector<std::pair<glm::vec3, glm::vec3>> chkPoints;

    float h_delta = (sphere_count/spheresGridWidth) * 2.0f / interleaveH;
    float w_delta = (float)(spheresGridWidth / interleaveW) * 2.0f;
    for (int i = 0; i < interleaveW; i++)
    {
        for (int j = 0; j < interleaveH; j++)
        {
            chkPoints.push_back(
                {
                    glm::vec3(w_delta*(float)i + 1.0f, h_delta*(float)j, w_delta*(float)i - 3.0f), 
                    glm::vec3(w_delta*(float)i, h_delta*(float)j, w_delta*(float)i)
                }
            );
        }
    }

    for (int i = 0; i < interleaveW; i++)
    {
        for (int j = 0; j < interleaveZ; j++)
        {
            chkPoints.push_back(
                {
                    glm::vec3(w_delta*(float)i + (float)j + 1.0f, h_delta*interleaveH/2.0f, w_delta*(float)i - (float)j - 1.0f), 
                    glm::vec3(w_delta*(float)i, h_delta*interleaveH/2.0f, w_delta*(float)i)
                }
            );
        }
    }
    return chkPoints;
}


std::vector<std::pair<glm::vec3, glm::vec3>> benchmark2_loaded_molecules(std::vector<glm::vec4>& spheres, int interleaveAngle, int interleaveZ, int interleaveY)
{
    float x_max = FLT_MIN;
    float y_max = FLT_MIN, y_min = FLT_MAX;
    float z_max = FLT_MIN;
    glm::vec3 mass_center(0.0f);

    for (auto& sphere : spheres)
    {
        mass_center += glm::vec3(sphere);
        if (sphere.y < y_min) y_min = sphere.y;
        if (sphere.y > y_max) y_max = sphere.y;
        if (abs(sphere.x) > x_max) x_max = abs(sphere.x);
        if (abs(sphere.z) > z_max) z_max = abs(sphere.z);
    }
    mass_center /= spheres.size();

    float d_theta = 360.0f/(float)interleaveAngle;
    float radius = sqrt(x_max*x_max + z_max*z_max);
    float d_radius = radius/(float) interleaveZ;
    float d_height = (y_max - y_min) / (float) interleaveY;

    std::vector<std::pair<glm::vec3, glm::vec3>> chkPoints;
    
    for (int i = 0; i < interleaveAngle; i++)
    {
        for (int j = 1; j < interleaveZ; j++)
        {
            for (int k = 0; k <= interleaveY; k++)
            {
                float d = (float)j * d_radius;
                chkPoints.push_back(
                    {
                        mass_center +
                        glm::vec3(d*cos(glm::radians((float)i*d_theta)), 
                        y_min + (float)k*d_height, 
                        d*sin(glm::radians((float)i*d_theta))), 
                        glm::vec3(mass_center.x, y_min + (float)k*d_height, mass_center.z)
                    }
                );
            }
        }
    }

    return chkPoints;
}

std::vector<std::pair<glm::vec3, glm::vec3>> benchmark3_package(int& sphere_count, int spheresGridWidth, int spheresGridHeight, int spheresGridDepth, int interleaveW, int interleaveH, int interleaveZ)
{
    float w_delta = (float)spheresGridWidth / interleaveW;
    float h_delta = (float)spheresGridHeight / interleaveH;
    float d_delta = (float)spheresGridDepth / interleaveZ;
    
    std::vector<std::pair<glm::vec3, glm::vec3>> chkPoints;

    for (int i = 0; i < interleaveW; i++)
    {
        for (int j = 0; j < interleaveH; j++)
        {
            for (int k = 0; k <= interleaveZ; k++)
            {
                float x = (float)i * w_delta;
                float y = (float)j * h_delta;
                float z = (float)k * d_delta;
                chkPoints.push_back(
                    {
                        glm::vec3(x, y, z),
                        glm::vec3(x, y, z + 1.0f) 
                    }
                );

            }
        }
    }

    for (int i = 0; i < interleaveW; i++)
    {
        for (int j = 0; j < interleaveH; j++)
        {
            for (int k = 0; k <= interleaveZ; k++)
            {
                if (i == j && j == k)
                {
                    chkPoints.push_back(
                        {
                            glm::vec3((float)i*2.0f - 10.0f, (float)j*2.0f - 10.0f, (float)k*2.0f - 10.0f),
                            glm::vec3((float)i*2.0f + 1.0f, (float)j*2.0f + 1.0f, (float)k*2.0f + 1.0f) 
                        }
                    );
                }
            }
        }
    }

    for (int k = 0; k <= interleaveZ; k++)
    {
        
        chkPoints.push_back(
            {
                glm::vec3((float)spheresGridWidth, (float)spheresGridHeight, -(float)k*2.0f),
                glm::vec3((float)spheresGridWidth, (float)spheresGridHeight, 1.0f) 
            }
        );
        
    }
    
    return chkPoints;

}


std::vector<glm::vec4> getScene(int sphere_count)
{
    switch (currentScene)
    {
    case SceneType::LOADED_SCENE:
        return loaded_scene(scene_path, sphere_count);
        break;
    case SceneType::GRID_SCENE:
        return grid_scene(spheresGridWidth, sphere_count);
        break;
    case SceneType::PACKAGE_SCENE:
        return package_scene(spheresGridWidth, spheresGridHeight, spheresGridDepth, sphere_count);
        break;
    default:
        return grid_scene(spheresGridWidth, sphere_count);
        break;
    }
}

std::vector<std::pair<glm::vec3, glm::vec3>> getCheckpoints(int& sphere_count, std::vector<glm::vec4>& spheres)
{
    switch (currentScene)
    {
    case SceneType::LOADED_SCENE:
        return benchmark2_loaded_molecules(spheres, interleaveAngle, interleaveZ, interleaveY);
        break;
    case SceneType::GRID_SCENE:
        return benchmark1_structured_grid(sphere_count, spheresGridWidth, interleaveW, interleaveH, interleaveZ);
        break;
    case SceneType::PACKAGE_SCENE:
        return benchmark3_package(sphere_count, spheresGridWidth, spheresGridHeight, spheresGridDepth, interleaveW, interleaveH, interleaveZ);
        break;
    default:
        return benchmark1_structured_grid(sphere_count, spheresGridWidth, interleaveW, interleaveH, interleaveZ);
        break;
    }
}

void takeScreenshot(SDL_Renderer* renderer, std::string& sshot_name)
{
    SDL_Surface* sshot = SDL_RenderReadPixels(renderer, nullptr);
    bool sshot_saved = SDL_SaveBMP(sshot, sshot_name.c_str());
    if (sshot_saved) std::cout << "Screenshot " << sshot_name << " saved." << std::endl;
    else std::cout << "Screenshot couldnt be saved." << std::endl;
    SDL_DestroySurface(sshot);
}