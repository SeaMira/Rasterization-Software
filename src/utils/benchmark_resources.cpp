#include <algorithm>
#include <cfloat>
#include <cmath>
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

namespace {

/** Exterior-only benchmark path for synthetic sphere clouds (grid sheet or 3D package). */
std::vector<std::pair<glm::vec3, glm::vec3>> exterior_synthetic_benchmark_checkpoints(
    const std::vector<glm::vec4>& spheres,
    bool package3D)
{
    std::vector<std::pair<glm::vec3, glm::vec3>> chkPoints;
    if (spheres.empty())
        return {{glm::vec3(0.0f), glm::vec3(0.0f)}};

    glm::vec3 com(0.0f);
    glm::vec3 minp(FLT_MAX);
    glm::vec3 maxp(-FLT_MAX);
    float maxR = 0.0f;
    for (const auto& s : spheres)
    {
        glm::vec3 c(s);
        float r = s.w;
        maxR = std::max(maxR, r);
        com += c;
        minp = glm::min(minp, c - glm::vec3(r));
        maxp = glm::max(maxp, c + glm::vec3(r));
    }
    com /= static_cast<float>(spheres.size());

    const int nAng = std::max(1, interleaveAngle);
    const int nRad = std::max(1, interleaveZ);
    const int nCorner = std::max(1, interleaveW);
    const int sliceMax = std::max(0, interleaveY);

    float x_dis_max = std::max(std::fabs(maxp.x - com.x), std::fabs(minp.x - com.x));
    float y_dis_max = std::max(std::fabs(maxp.y - com.y), std::fabs(minp.y - com.y));
    float z_dis_max = std::max(std::fabs(maxp.z - com.z), std::fabs(minp.z - com.z));

    auto orbit_radius = [&](float a, float b, float rf) -> float {
        float R = std::sqrt(a * a + b * b) * rf;
        if (R < 1e-4f)
            R = std::max(maxR * 4.0f, 1.0f);
        return R;
    };

    const float d_theta = 360.0f / static_cast<float>(nAng);

    // Orbit in XZ, multiple Y slices (works for flat grid and 3D)
    {
        float orbitRBase = orbit_radius(x_dis_max, z_dis_max, radFactor);
        float d_radius = orbitRBase / static_cast<float>(nRad);
        float y_min = minp.y;
        float y_max = maxp.y;
        float d_height = (sliceMax > 0 && y_max > y_min) ? (y_max - y_min) / static_cast<float>(sliceMax) : 0.0f;
        for (int j = 1; j <= nRad; ++j)
        {
            float d = static_cast<float>(j) * d_radius;
            for (int i = 0; i < nAng; ++i)
            {
                float ang = glm::radians(static_cast<float>(i) * d_theta);
                float c0 = std::cos(ang);
                float s0 = std::sin(ang);
                for (int k = 0; k <= sliceMax; ++k)
                {
                    float y = (sliceMax == 0 || y_max <= y_min) ? com.y : (y_min + static_cast<float>(k) * d_height);
                    chkPoints.emplace_back(
                        glm::vec3(com.x + d * c0, y, com.z + d * s0),
                        glm::vec3(com.x, y, com.z));
                }
            }
        }
    }

    if (package3D)
    {
        // Orbit in XY, multiple Z slices
        {
            float orbitRBase = orbit_radius(x_dis_max, y_dis_max, radFactor);
            float d_radius = orbitRBase / static_cast<float>(nRad);
            float z_min = minp.z;
            float z_max = maxp.z;
            float d_depth = (sliceMax > 0 && z_max > z_min) ? (z_max - z_min) / static_cast<float>(sliceMax) : 0.0f;
            for (int j = 1; j <= nRad; ++j)
            {
                float d = static_cast<float>(j) * d_radius;
                for (int i = 0; i < nAng; ++i)
                {
                    float ang = glm::radians(static_cast<float>(i) * d_theta);
                    float c0 = std::cos(ang);
                    float s0 = std::sin(ang);
                    for (int k = 0; k <= sliceMax; ++k)
                    {
                        float z = (sliceMax == 0 || z_max <= z_min) ? com.z : (z_min + static_cast<float>(k) * d_depth);
                        chkPoints.emplace_back(
                            glm::vec3(com.x + d * c0, com.y + d * s0, z),
                            glm::vec3(com.x, com.y, z));
                    }
                }
            }
        }
        // Orbit in YZ, multiple X slices
        {
            float orbitRBase = orbit_radius(y_dis_max, z_dis_max, radFactor);
            float d_radius = orbitRBase / static_cast<float>(nRad);
            float x_min = minp.x;
            float x_max = maxp.x;
            float d_width = (sliceMax > 0 && x_max > x_min) ? (x_max - x_min) / static_cast<float>(sliceMax) : 0.0f;
            for (int j = 1; j <= nRad; ++j)
            {
                float d = static_cast<float>(j) * d_radius;
                for (int i = 0; i < nAng; ++i)
                {
                    float ang = glm::radians(static_cast<float>(i) * d_theta);
                    float c0 = std::cos(ang);
                    float s0 = std::sin(ang);
                    for (int k = 0; k <= sliceMax; ++k)
                    {
                        float x = (sliceMax == 0 || x_max <= x_min) ? com.x : (x_min + static_cast<float>(k) * d_width);
                        chkPoints.emplace_back(
                            glm::vec3(x, com.y + d * c0, com.z + d * s0),
                            glm::vec3(x, com.y, com.z));
                    }
                }
            }
        }
    }

    glm::vec3 diag = maxp - minp;
    float diagonal = glm::length(diag);
    if (diagonal < 1e-6f)
        diagonal = std::max(maxR * 4.0f, 1.0f);

    const float clearance = std::max(maxR * 2.5f, 0.5f);
    const float farExtra = diagonal * std::max(radFactor, 1.0f);

    for (int ix = 0; ix < 2; ++ix)
    {
        for (int iy = 0; iy < 2; ++iy)
        {
            for (int iz = 0; iz < 2; ++iz)
            {
                glm::vec3 corner(
                    ix ? maxp.x : minp.x,
                    iy ? maxp.y : minp.y,
                    iz ? maxp.z : minp.z);
                glm::vec3 toCorner = corner - com;
                float len = glm::length(toCorner);
                if (len < 1e-5f)
                    continue;
                glm::vec3 u = toCorner / len;

                for (int s = 0; s < nCorner; ++s)
                {
                    float alpha = (nCorner == 1) ? 0.5f : static_cast<float>(s) / static_cast<float>(nCorner - 1);
                    float tNear = len + clearance;
                    float tFar = len + clearance + farExtra;
                    float t = tNear + alpha * (tFar - tNear);
                    chkPoints.emplace_back(com + u * t, com);
                }
            }
        }
    }

    return chkPoints;
}

} // namespace

std::vector<std::pair<glm::vec3, glm::vec3>> benchmark1_structured_grid(const std::vector<glm::vec4>& spheres)
{
    return exterior_synthetic_benchmark_checkpoints(spheres, false);
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

std::vector<std::pair<glm::vec3, glm::vec3>> benchmark3_package(const std::vector<glm::vec4>& spheres)
{
    return exterior_synthetic_benchmark_checkpoints(spheres, true);
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
        return benchmark1_structured_grid(spheres);
        break;
    case SceneType::PACKAGE_SCENE:
        return benchmark3_package(spheres);
        break;
    default:
        return benchmark1_structured_grid(spheres);
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
        return benchmark1_structured_grid(spheres);
        break;
    case SceneType::PACKAGE_SCENE:
        return benchmark3_package(spheres);
        break;
    default:
        return benchmark1_structured_grid(spheres);
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