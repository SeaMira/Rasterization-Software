#include <iostream>
#include "utils/benchmark_resources.h"
#include "molecule_loader/basic_loader.h"

// LOADED_SCENE or GRID_SCENE
SceneType currentScene = SceneType::GRID_SCENE;
std::string scene_file = "8wql.cif";
std::filesystem::path scene_path = std::filesystem::path("assets/molecules") / scene_file;

int gridWidth = 10;
int gridHeight = 10;
int gridDepth = 10;
int interleaveW = 5;
int interleaveH = 5;

int interleaveAngle = 8;
int interleaveZ = 5;
int interleaveY = 5;
// interleaveAngle*interleaveZ*(interleaveY+1) checkpoints will be created
float radFactor = 2.0f;
float separation = 2.0f;

float sphereRadius = 0.5f;
float cylinderRadius = 0.2f;

std::vector<glm::vec4> loaded_scene(std::filesystem::path& path, int sphere_count)
{
    ChemFilesLoader loader(path);
    std::vector<glm::vec4>& positions = loader.getSphereInfo();
    std::vector<glm::vec4> spheres(positions.begin(), positions.begin() + std::min(positions.size(), static_cast<size_t>(sphere_count)));
    return spheres;
}

std::vector<CylinderIndex> loaded_cylinder_scene(std::filesystem::path& path, int cylinder_count)
{
    ChemFilesLoader loader(path);
    std::vector<CylinderIndex> cylinders;
    auto& bonds = loader.getBondsInfo();
    for (int i = 0; i < std::min(loader.getBondsAmount(), cylinder_count); i++)
    {
        const auto& bond = bonds[i];
        cylinders.push_back(CylinderIndex(bond.first, bond.second, cylinderRadius));
    }
    return cylinders;
}

void loaded_complete_scene(std::filesystem::path& path, std::vector<glm::vec4>& spheres, int sphere_count, std::vector<CylinderIndex>& cylinders, int cylinder_count)
{
    ChemFilesLoader loader(path);
    std::vector<glm::vec4>& positions = loader.getSphereInfo();
    spheres.assign(positions.begin(), positions.begin() + std::min(positions.size(), static_cast<size_t>(sphere_count)));

    cylinders.clear();
    auto& bonds = loader.getBondsInfo();
    for (int i = 0; i < std::min(loader.getBondsAmount(), cylinder_count); i++)
    {
        const auto& bond = bonds[i];
        cylinders.push_back(CylinderIndex(bond.first, bond.second, cylinderRadius));
    }
}

std::vector<glm::vec4> grid_scene(int gridWidth, int sphere_count)
{
    std::vector<glm::vec4> spheres;
    for (int i = 0; i < sphere_count; i++)
    {
        spheres.push_back({(float)(i%gridWidth)*separation, (float)(i/gridWidth) * separation, (float)(i%gridWidth)*separation, sphereRadius});
    }
    return spheres;
}

std::vector<CylinderIndex> cylinder_grid_scene(int gridWidth, int cylinder_count)
{
    std::vector<CylinderIndex> cylinders;
    for (int i = 0; i < cylinder_count; i++)
    {
        // Connect sphere i to sphere i+1
        cylinders.push_back(CylinderIndex(i, i + 1, cylinderRadius));
    }
    return cylinders;
}

void complete_grid_scene(int gridWidth, std::vector<glm::vec4>& spheres, int sphere_count, std::vector<CylinderIndex>& cylinders, int cylinder_count)
{
    spheres.clear();
    cylinders.clear();

    spheres.reserve(sphere_count);
    cylinders.reserve(cylinder_count);

    int n_entities = std::max(sphere_count, cylinder_count);
    int sn_entities = std::sqrt(n_entities) + 1;

    for (int i = 0; i < sphere_count; i++)
    {
        spheres.push_back({(float)(i%sn_entities)*separation, (float)(i/sn_entities) * separation, 1.0f, sphereRadius});
    }
    for (int i = 0; i < cylinder_count; i++)
    {
        // Connect sphere i to sphere i+1 (clamped to valid range)
        uint32_t idxA = i % sphere_count;
        uint32_t idxB = (i + 1) % sphere_count;
        cylinders.push_back(CylinderIndex(idxA, idxB, cylinderRadius));
    }
}

std::vector<glm::vec4> package_scene(int gridWidth, int gridHeight, int gridDepth, int sphere_count)
{
    std::vector<glm::vec4> spheres;
    int count = 0;
    for (int i = 0; i < gridWidth; i++)
    {
        for (int j = 0; j < gridHeight; j++)
        {
            for (int k = 0; k < gridDepth; k++)
            {
                if (count >= sphere_count) break;
                spheres.push_back({(float)i*2.0f, (float)j*2.0f, (float)k*2.0f, 1.0f});
                count++;
            }
        }
    }
    return spheres;
}

std::vector<CylinderIndex> cylinder_package_scene(int gridWidth, int gridHeight, int gridDepth, int cylinder_count)
{
    std::vector<CylinderIndex> cylinders;
    int count = 0;
    for (int i = 0; i < gridWidth; i++)
    {
        for (int j = 0; j < gridHeight; j++)
        {
            for (int k = 0; k < gridDepth; k++)
            {
                if (count >= cylinder_count) return cylinders;
                // Connect sphere at current index to next index
                int currentIdx = i * gridHeight * gridDepth + j * gridDepth + k;
                int nextIdx = i * gridHeight * gridDepth + j * gridDepth + k + 1;
                if (k + 1 < gridDepth)
                    cylinders.push_back(CylinderIndex(currentIdx, nextIdx, .5f));
                else
                    cylinders.push_back(CylinderIndex(currentIdx, currentIdx, .5f));
                count++;
            }
        }
    }
    return cylinders;
}

void complete_package_scene(int gridWidth, int gridHeight, int gridDepth, std::vector<glm::vec4>& spheres, int sphere_count, std::vector<CylinderIndex>& cylinders, int cylinder_count)
{
    int s_count = 0;
    int c_count = 0;
    spheres.clear();
    cylinders.clear();
    spheres.reserve(sphere_count);
    cylinders.reserve(cylinder_count);
    for (int i = 0; i < gridWidth; i++)
    {
        for (int j = 0; j < gridHeight; j++)
        {
            for (int k = 0; k < gridDepth; k++)
            {
                if (s_count < sphere_count) 
                {
                    spheres.push_back({(float)i*2.0f, (float)j*2.0f, (float)k*2.0f, 1.0f});
                    s_count++;
                }

                if (c_count < cylinder_count)
                {
                    int currentIdx = i * gridHeight * gridDepth + j * gridDepth + k;
                    int nextIdx = (k + 1 < gridDepth) ? (currentIdx + 1) : currentIdx;
                    cylinders.push_back(CylinderIndex(currentIdx, nextIdx, .5f));
                    c_count++;
                }
            }
        }
    }
}

std::vector<std::pair<glm::vec3, glm::vec3>> benchmark1_structured_grid(int& sphere_count, int gridWidth, int interleaveW, int interleaveH, int interleaveZ)
{
    std::vector<std::pair<glm::vec3, glm::vec3>> chkPoints;

    float h_delta = (sphere_count/gridWidth) * 2.0f / interleaveH;
    float w_delta = (float)(gridWidth / interleaveW) * 2.0f;
    for (int i = 0; i < interleaveW; i++)
    {
        for (int j = 0; j < interleaveH; j++)
        {
            chkPoints.push_back(
                {
                    glm::vec3(w_delta*(float)i, h_delta*(float)j, - 3.0f), 
                    glm::vec3(w_delta*(float)i, h_delta*(float)j, 1.0f)
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
                    glm::vec3(w_delta*(float)i, h_delta*interleaveH/2.0f,  - (float)j - 1.0f), 
                    glm::vec3(w_delta*(float)i, h_delta*interleaveH/2.0f, 0.0f)
                }
            );
        }
    }

    int sn_entities = std::sqrt(sphere_count);
    for (int j = 0; j < sn_entities; j++)
    {
        chkPoints.push_back(
            {
                glm::vec3(separation*(sn_entities/2), separation*(sn_entities/2),  - (float)j*separation - 1.0f), 
                glm::vec3(separation*(sn_entities/2), separation*(sn_entities/2), 0.0f)
            }
        );
    }

    return chkPoints;
}


std::vector<std::pair<glm::vec3, glm::vec3>> benchmark2_loaded_molecules(std::vector<glm::vec4>& spheres, std::vector<CylinderIndex>& cylinders, int interleaveAngle, int interleaveZ, int interleaveY, float radFactor)
{
    float x_dis_max;
    float z_dis_max;
    glm::vec3 mass_center(0.0f);
    glm::vec3 min_point(FLT_MAX);
    glm::vec3 max_point(FLT_MIN);

    if (spheres.empty() && cylinders.empty()) return {{mass_center, mass_center}};

    // Always compute from spheres (cylinder endpoints are sphere positions)
    for (auto& sphere : spheres)
    {
        mass_center += glm::vec3(sphere);
        min_point = glm::min(min_point, glm::vec3(sphere));
        max_point = glm::max(max_point, glm::vec3(sphere));
    }
    if (!spheres.empty()) mass_center /= spheres.size();

    float d_theta = 360.0f/(float)interleaveAngle;
    x_dis_max = glm::max(abs(max_point.x - mass_center.x), abs(min_point.x - mass_center.x));
    z_dis_max = glm::max(abs(max_point.z - mass_center.z), abs(min_point.z - mass_center.z));
    float radius = sqrt(x_dis_max*x_dis_max + z_dis_max*z_dis_max) * radFactor;
    float d_radius = radius/(float) interleaveZ;
    float d_height = (max_point.y - min_point.y) / (float) interleaveY;

    std::vector<std::pair<glm::vec3, glm::vec3>> chkPoints;
    
    for (int i = 1; i <= interleaveZ; i++)
    {
        float d = (float)i * d_radius;
        for (int j = 0; j < interleaveAngle; j++)
        {
            for (int k = 0; k <= interleaveY; k++)
            {
                chkPoints.push_back(
                    {
                        mass_center +
                        glm::vec3(d*cos(glm::radians((float)j*d_theta)), 
                        min_point.y + (float)k*d_height, 
                        d*sin(glm::radians((float)j*d_theta))), 
                        glm::vec3(mass_center.x, min_point.y + (float)k*d_height, mass_center.z)
                    }
                );
            }
        }
    }

    return chkPoints;
}

std::vector<std::pair<glm::vec3, glm::vec3>> benchmark2_loaded_molecules(std::vector<glm::vec4>& spheres, int interleaveAngle, int interleaveZ, int interleaveY, float radFactor)
{
    float x_dis_max;
    float z_dis_max;

    glm::vec3 mass_center(0.0f);
    glm::vec3 min_point(FLT_MAX);
    glm::vec3 max_point(FLT_MIN);

    if (spheres.empty()) return {{mass_center, mass_center}};

    
    for (auto& sphere : spheres)
    {
        mass_center += glm::vec3(sphere);
        min_point = glm::min(min_point, glm::vec3(sphere));
        max_point = glm::max(max_point, glm::vec3(sphere));
    }
    mass_center /= spheres.size();
    

    float d_theta = 360.0f/(float)interleaveAngle;
    x_dis_max = glm::max(abs(max_point.x - mass_center.x), abs(min_point.x - mass_center.x));
    z_dis_max = glm::max(abs(max_point.z - mass_center.z), abs(min_point.z - mass_center.z));
    float radius = sqrt(x_dis_max*x_dis_max + z_dis_max*z_dis_max) * radFactor;
    float d_radius = radius/(float) interleaveZ;
    float d_height = (max_point.y - min_point.y) / (float) interleaveY;

    std::vector<std::pair<glm::vec3, glm::vec3>> chkPoints;
    
    for (int j = 1; j <= interleaveZ; j++)
    {
        for (int i = 0; i < interleaveAngle; i++)
        {
            for (int k = 0; k <= interleaveY; k++)
            {
                float d = (float)j * d_radius;
                chkPoints.push_back(
                    {
                        mass_center +
                        glm::vec3(d*cos(glm::radians((float)i*d_theta)), 
                        min_point.y + (float)k*d_height, 
                        d*sin(glm::radians((float)i*d_theta))), 
                        glm::vec3(mass_center.x, min_point.y + (float)k*d_height, mass_center.z)
                    }
                );
            }
        }
    }

    return chkPoints;
}

std::vector<std::pair<glm::vec3, glm::vec3>> benchmark3_package(int& sphere_count, int gridWidth, int gridHeight, int gridDepth, int interleaveW, int interleaveH, int interleaveZ)
{
    float w_delta = (float)gridWidth / interleaveW;
    float h_delta = (float)gridHeight / interleaveH;
    float d_delta = (float)gridDepth / interleaveZ;
    
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
                glm::vec3((float)gridWidth, (float)gridHeight, -(float)k*2.0f),
                glm::vec3((float)gridWidth, (float)gridHeight, 1.0f) 
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
        return grid_scene(gridWidth, sphere_count);
        break;
    case SceneType::PACKAGE_SCENE:
        return package_scene(gridWidth, gridHeight, gridDepth, sphere_count);
        break;
    default:
        return grid_scene(gridWidth, sphere_count);
        break;
    }
}

std::vector<CylinderIndex> getCylinderScene(int cylinder_count)
{
    switch (currentScene)
    {
    case SceneType::LOADED_SCENE:
        return loaded_cylinder_scene(scene_path, cylinder_count);
        break;
    case SceneType::GRID_SCENE:
        return cylinder_grid_scene(gridWidth, cylinder_count);
        break;
    case SceneType::PACKAGE_SCENE:
        return cylinder_package_scene(gridWidth, gridHeight, gridDepth, cylinder_count);
        break;
    default:
        return cylinder_grid_scene(gridWidth, cylinder_count);
        break;
    }
}


void getCompleteScene(std::vector<glm::vec4>& spheres, int sphere_count, std::vector<CylinderIndex>& cylinders, int cylinder_count)
{
    switch (currentScene)
    {
    case SceneType::LOADED_SCENE:
        loaded_complete_scene(scene_path, spheres, sphere_count, cylinders, cylinder_count);
        break;
    case SceneType::GRID_SCENE:
        complete_grid_scene(gridWidth, spheres, sphere_count, cylinders, cylinder_count);
        break;
    case SceneType::PACKAGE_SCENE:
        complete_package_scene(gridWidth, gridHeight, gridDepth, spheres, sphere_count, cylinders, cylinder_count);
        break;
    default:
        complete_grid_scene(gridWidth, spheres, sphere_count, cylinders, cylinder_count);
        break;
    }
}

std::vector<std::pair<glm::vec3, glm::vec3>> getCheckpoints(int& sphere_count, std::vector<glm::vec4>& spheres, std::vector<CylinderIndex>& cylinders)
{
    switch (currentScene)
    {
    case SceneType::LOADED_SCENE:
        return benchmark2_loaded_molecules(spheres, cylinders, interleaveAngle, interleaveZ, interleaveY, radFactor);
        break;
    case SceneType::GRID_SCENE:
        return benchmark1_structured_grid(sphere_count, gridWidth, interleaveW, interleaveH, interleaveZ);
        break;
    case SceneType::PACKAGE_SCENE:
        return benchmark3_package(sphere_count, gridWidth, gridHeight, gridDepth, interleaveW, interleaveH, interleaveZ);
        break;
    default:
        return benchmark1_structured_grid(sphere_count, gridWidth, interleaveW, interleaveH, interleaveZ);
        break;
    }
}

std::vector<std::pair<glm::vec3, glm::vec3>> getCheckpoints(int& sphere_count, std::vector<glm::vec4>& spheres)
{
    switch (currentScene)
    {
    case SceneType::LOADED_SCENE:
        return benchmark2_loaded_molecules(spheres, interleaveAngle, interleaveZ, interleaveY, radFactor);
        break;
    case SceneType::GRID_SCENE:
        return benchmark1_structured_grid(sphere_count, gridWidth, interleaveW, interleaveH, interleaveZ);
        break;
    case SceneType::PACKAGE_SCENE:
        return benchmark3_package(sphere_count, gridWidth, gridHeight, gridDepth, interleaveW, interleaveH, interleaveZ);
        break;
    default:
        return benchmark1_structured_grid(sphere_count, gridWidth, interleaveW, interleaveH, interleaveZ);
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