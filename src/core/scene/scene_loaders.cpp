#include "core/scene/scene_loaders.h"
#include "core/scene/scene.h"
#include "utils/benchmark_resources.h"

#include <stdexcept>
#include <algorithm>

// ============ FileSceneLoader ============

FileSceneLoader::FileSceneLoader(const SceneConfig& config)
    : m_config(config)
{
}

std::unique_ptr<Scene> FileSceneLoader::load()
{
    auto scene = std::make_unique<Scene>(m_config);
    
    std::filesystem::path path = m_config.getFilePath();
    
    // Load spheres and cylinders using existing utility function
    std::vector<glm::vec4> sphereData;
    std::vector<CylinderIndex> cylinders;
    
    loaded_complete_scene(
        path,
        sphereData,
        m_config.getSphereCount(),
        cylinders,
        m_config.getCylinderCount()
    );
    
    // Add to scene
    scene->addSpheresFromVec4(sphereData);
    scene->addCylinders(cylinders);
    
    return scene;
}

bool FileSceneLoader::canLoad(const std::filesystem::path& path) const
{
    static const std::vector<std::string> supportedExtensions = {
        ".mmtf", ".pdb", ".mol2", ".xyz", ".cif"
    };
    
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    return std::find(supportedExtensions.begin(), supportedExtensions.end(), ext) 
           != supportedExtensions.end();
}

// ============ GridSceneLoader ============

GridSceneLoader::GridSceneLoader(const SceneConfig& config)
    : m_config(config)
{
}

std::unique_ptr<Scene> GridSceneLoader::load()
{
    auto scene = std::make_unique<Scene>(m_config);
    
    generateSpheres(*scene);
    generateCylinders(*scene);
    
    return scene;
}

void GridSceneLoader::generateSpheres(Scene& scene) const
{
    int gridW = m_config.getGridWidth();
    int gridH = m_config.getGridHeight();
    int gridD = m_config.getGridDepth();
    float separation = m_config.getSeparation();
    float radius = m_config.getSphereRadius();
    
    scene.reserveSpheres(static_cast<size_t>(gridW * gridH * gridD));
    
    for (int x = 0; x < gridW; ++x)
    {
        for (int y = 0; y < gridH; ++y)
        {
            for (int z = 0; z < gridD; ++z)
            {
                Sphere sphere;
                sphere.m_position = glm::vec3(
                    static_cast<float>(x) * separation,
                    static_cast<float>(y) * separation,
                    static_cast<float>(z) * separation
                );
                sphere.m_radius = radius;
                scene.addSphere(sphere);
            }
        }
    }
}

void GridSceneLoader::generateCylinders(Scene& scene) const
{
    int gridW = m_config.getGridWidth();
    int gridH = m_config.getGridHeight();
    int gridD = m_config.getGridDepth();
    float separation = m_config.getSeparation();
    float radius = m_config.getCylinderRadius();
    
    auto getIndex = [gridH, gridD](int x, int y, int z) {
        return x * gridH * gridD + y * gridD + z;
    };
    
    const auto& spheres = scene.getSpheres();
    
    // Connect adjacent spheres with cylinders
    for (int x = 0; x < gridW; ++x)
    {
        for (int y = 0; y < gridH; ++y)
        {
            for (int z = 0; z < gridD; ++z)
            {
                int currentIdx = getIndex(x, y, z);
                
                // Connect to neighbor in X direction
                if (x < gridW - 1)
                {
                    int neighborIdx = getIndex(x + 1, y, z);
                    scene.addCylinder(CylinderIndex(
                        static_cast<uint32_t>(currentIdx),
                        static_cast<uint32_t>(neighborIdx),
                        radius
                    ));
                }
                
                // Connect to neighbor in Y direction
                if (y < gridH - 1)
                {
                    int neighborIdx = getIndex(x, y + 1, z);
                    scene.addCylinder(CylinderIndex(
                        static_cast<uint32_t>(currentIdx),
                        static_cast<uint32_t>(neighborIdx),
                        radius
                    ));
                }
                
                // Connect to neighbor in Z direction
                if (z < gridD - 1)
                {
                    int neighborIdx = getIndex(x, y, z + 1);
                    scene.addCylinder(CylinderIndex(
                        static_cast<uint32_t>(currentIdx),
                        static_cast<uint32_t>(neighborIdx),
                        radius
                    ));
                }
            }
        }
    }
}

// ============ SceneLoaderFactory ============

std::unique_ptr<ISceneLoader> SceneLoaderFactory::create(const SceneConfig& config)
{
    switch (config.getSourceType())
    {
        case SceneSourceType::FILE_LOADED:
            return std::make_unique<FileSceneLoader>(config);
            
        case SceneSourceType::GRID_GENERATED:
            return std::make_unique<GridSceneLoader>(config);
            
        case SceneSourceType::PACKED_SCENE:
            // For packed scenes, use grid loader with custom parameters
            return std::make_unique<GridSceneLoader>(config);
            
        default:
            throw std::invalid_argument("Unknown scene source type");
    }
}
