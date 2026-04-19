#include "utils/scene_config_loader.h"
#include "utils/benchmark_resources.h"
#include <sstream>
#include <algorithm>
#include <cctype>

// Helper function to trim whitespace
static std::string trim(const std::string& str) 
{
    size_t first = str.find_first_not_of(" \t\n\r\"");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\n\r\"");
    return str.substr(first, (last - first + 1));
}

// Helper function to find a value in JSON by key
static std::string findJsonValue(const std::string& json, const std::string& key) 
{
    std::string searchKey = "\"" + key + "\"";
    size_t keyPos = json.find(searchKey);
    if (keyPos == std::string::npos) return "";

    size_t colonPos = json.find(':', keyPos);
    if (colonPos == std::string::npos) return "";

    size_t valueStart = colonPos + 1;
    
    // Skip whitespace
    while (valueStart < json.length() && std::isspace(json[valueStart])) 
    {
        valueStart++;
    }

    if (valueStart >= json.length()) return "";

    // Check if value is a string (starts with quote)
    if (json[valueStart] == '"') 
    {
        size_t valueEnd = json.find('"', valueStart + 1);
        if (valueEnd == std::string::npos) return "";
        return json.substr(valueStart + 1, valueEnd - valueStart - 1);
    }
    
    // Check if value is an object or array
    if (json[valueStart] == '{' || json[valueStart] == '[') 
    {
        char openBracket = json[valueStart];
        char closeBracket = (openBracket == '{') ? '}' : ']';
        int depth = 1;
        size_t valueEnd = valueStart + 1;
        while (valueEnd < json.length() && depth > 0) 
        {
            if (json[valueEnd] == openBracket) depth++;
            else if (json[valueEnd] == closeBracket) depth--;
            valueEnd++;
        }
        return json.substr(valueStart, valueEnd - valueStart);
    }

    // Value is a number or boolean
    size_t valueEnd = valueStart;
    while (valueEnd < json.length() && 
           json[valueEnd] != ',' && 
           json[valueEnd] != '}' && 
           json[valueEnd] != ']' &&
           !std::isspace(json[valueEnd])) 
    {
        valueEnd++;
    }
    
    return trim(json.substr(valueStart, valueEnd - valueStart));
}

std::string SceneConfigLoader::parseString(const std::string& json, const std::string& key, const std::string& defaultValue) 
{
    std::string value = findJsonValue(json, key);
    return value.empty() ? defaultValue : value;
}

int SceneConfigLoader::parseInt(const std::string& json, const std::string& key, int defaultValue) 
{
    std::string value = findJsonValue(json, key);
    if (value.empty()) return defaultValue;
    try 
    {
        return std::stoi(value);
    } 
    catch (...) 
    {
        return defaultValue;
    }
}

float SceneConfigLoader::parseFloat(const std::string& json, const std::string& key, float defaultValue) 
{
    std::string value = findJsonValue(json, key);
    if (value.empty()) return defaultValue;
    try 
    {
        return std::stof(value);
    } 
    catch (...) 
    {
        return defaultValue;
    }
}

bool SceneConfigLoader::parseBool(const std::string& json, const std::string& key, bool defaultValue) 
{
    std::string value = findJsonValue(json, key);
    if (value.empty()) return defaultValue;
    
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    return value == "true" || value == "1";
}

SceneSettings SceneConfigLoader::loadFromFile(const std::string& configPath) 
{
    SceneSettings settings;

    std::ifstream file(configPath);
    if (!file.is_open()) 
    {
        std::cerr << "Warning: Could not open config file: " << configPath 
                  << ". Using default values." << std::endl;
        return settings;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json = buffer.str();
    file.close();

    std::cout << "Loading configuration from: " << configPath << std::endl;

    // Parse scene section
    std::string sceneSection = findJsonValue(json, "scene");
    if (!sceneSection.empty()) 
    {
        settings.sceneType = parseString(sceneSection, "type", settings.sceneType);
        settings.sceneFile = parseString(sceneSection, "file", settings.sceneFile);
        settings.sphereCount = parseInt(sceneSection, "sphere_count", settings.sphereCount);
        settings.cylinderCount = parseInt(sceneSection, "cylinder_count", settings.cylinderCount);
    }

    // Parse grid section
    std::string gridSection = findJsonValue(json, "grid");
    if (!gridSection.empty()) 
    {
        settings.gridWidth = parseInt(gridSection, "width", settings.gridWidth);
        settings.gridHeight = parseInt(gridSection, "height", settings.gridHeight);
        settings.gridDepth = parseInt(gridSection, "depth", settings.gridDepth);
    }

    // Parse benchmark section
    std::string benchmarkSection = findJsonValue(json, "benchmark");
    if (!benchmarkSection.empty()) 
    {
        settings.interleaveWidth = parseInt(benchmarkSection, "interleave_width", settings.interleaveWidth);
        settings.interleaveHeight = parseInt(benchmarkSection, "interleave_height", settings.interleaveHeight);
        settings.interleaveAngle = parseInt(benchmarkSection, "interleave_angle", settings.interleaveAngle);
        settings.interleaveZ = parseInt(benchmarkSection, "interleave_z", settings.interleaveZ);
        settings.interleaveY = parseInt(benchmarkSection, "interleave_y", settings.interleaveY);
        settings.radiusFactor = parseFloat(benchmarkSection, "radius_factor", settings.radiusFactor);
    }

    // Parse geometry section
    std::string geometrySection = findJsonValue(json, "geometry");
    if (!geometrySection.empty()) 
    {
        settings.separation = parseFloat(geometrySection, "separation", settings.separation);
        settings.sphereRadius = parseFloat(geometrySection, "sphere_radius", settings.sphereRadius);
        settings.cylinderRadius = parseFloat(geometrySection, "cylinder_radius", settings.cylinderRadius);
    }

    // Parse render section
    std::string renderSection = findJsonValue(json, "render");
    if (!renderSection.empty()) 
    {
        settings.screenWidth = parseInt(renderSection, "screen_width", settings.screenWidth);
        settings.screenHeight = parseInt(renderSection, "screen_height", settings.screenHeight);
        settings.withOcclusionCulling = parseBool(renderSection, "with_occlusion_culling", settings.withOcclusionCulling);
        settings.downsampleLevel = parseInt(renderSection, "downsample_level", settings.downsampleLevel);
        settings.workGroupSizePerPixelX = parseInt(renderSection, "work_group_size_per_pixel_x", settings.workGroupSizePerPixelX);
        settings.workGroupSizePerPixelY = parseInt(renderSection, "work_group_size_per_pixel_y", settings.workGroupSizePerPixelY);
        settings.workGroupSizePerSphere = parseInt(renderSection, "work_group_size_per_sphere", settings.workGroupSizePerSphere);
        settings.workGroupSizePerCylinder = parseInt(renderSection, "work_group_size_per_cylinder", settings.workGroupSizePerCylinder);
    }

    // Parse profiler section
    std::string profilerSection = findJsonValue(json, "profiler");
    if (!profilerSection.empty()) 
    {
        settings.timerDuration = parseFloat(profilerSection, "timer_duration", settings.timerDuration);
        settings.benchmarkingEnabled = parseBool(profilerSection, "benchmarking_enabled", settings.benchmarkingEnabled);
    }

    // Parse out-of-core section
    std::string oocSection = findJsonValue(json, "outofcore");
    if (!oocSection.empty()) 
    {
        settings.oocAtomsPerBlock       = parseInt(oocSection, "atoms_per_block", settings.oocAtomsPerBlock);
        settings.oocMaxOctreeDepth      = parseInt(oocSection, "max_octree_depth", settings.oocMaxOctreeDepth);
        settings.oocBlocksPerLeaf       = parseInt(oocSection, "blocks_per_leaf", settings.oocBlocksPerLeaf);
        settings.oocMaxBlockPoolSlots   = parseInt(oocSection, "max_block_pool_slots", settings.oocMaxBlockPoolSlots);
        settings.oocMaxRequestsPerFrame = parseInt(oocSection, "max_requests_per_frame", settings.oocMaxRequestsPerFrame);
        settings.oocVisibilityThreshold = parseFloat(oocSection, "visibility_threshold", settings.oocVisibilityThreshold);
        settings.oocOcclusionMethod     = parseString(oocSection, "occlusion_method", settings.oocOcclusionMethod);
        settings.oocStatsAccumulateFrames = parseInt(oocSection, "stats_accumulate_frames", settings.oocStatsAccumulateFrames);
        settings.oocStatsCsvPath          = parseString(oocSection, "stats_csv_path", settings.oocStatsCsvPath);
        settings.oocPreprocessStatsCsvPath = parseString(oocSection, "preprocess_stats_csv_path", settings.oocPreprocessStatsCsvPath);
    }

    std::cout << "Configuration loaded successfully." << std::endl;
    std::cout << "  Scene type: " << settings.sceneType << std::endl;
    std::cout << "  Scene file: " << settings.sceneFile << std::endl;
    std::cout << "  Sphere count: " << settings.sphereCount << std::endl;
    std::cout << "  Cylinder count: " << settings.cylinderCount << std::endl;

    return settings;
}

SceneSettings SceneConfigLoader::loadDefault() 
{
    return loadFromFile("assets/config/scene_config.json");
}

void SceneConfigLoader::applyToGlobals(const SceneSettings& settings) 
{
    // Apply scene type
    if (settings.sceneType == "LOADED_SCENE") 
    {
        currentScene = SceneType::LOADED_SCENE;
    } 
    else if (settings.sceneType == "GRID_SCENE") 
    {
        currentScene = SceneType::GRID_SCENE;
    } 
    else if (settings.sceneType == "PACKAGE_SCENE") 
    {
        currentScene = SceneType::PACKAGE_SCENE;
    }

    // Apply scene file
    scene_file = settings.sceneFile;
    scene_path = settings.getScenePath();

    // Apply grid settings
    gridWidth = settings.gridWidth;
    gridHeight = settings.gridHeight;
    gridDepth = settings.gridDepth;

    // Apply benchmark settings
    interleaveW = settings.interleaveWidth;
    interleaveH = settings.interleaveHeight;
    interleaveAngle = settings.interleaveAngle;
    interleaveZ = settings.interleaveZ;
    interleaveY = settings.interleaveY;
    radFactor = settings.radiusFactor;

    // Apply geometry settings
    separation = settings.separation;
    sphereRadius = settings.sphereRadius;
    cylinderRadius = settings.cylinderRadius;

    std::cout << "Global variables updated from configuration." << std::endl;
}

void SceneConfigLoader::saveToFile(const SceneSettings& settings, const std::string& configPath) 
{
    std::ofstream file(configPath);
    if (!file.is_open()) 
    {
        throw std::runtime_error("Could not open file for writing: " + configPath);
    }

    file << "{\n";
    
    // Scene section
    file << "    \"scene\": {\n";
    file << "        \"type\": \"" << settings.sceneType << "\",\n";
    file << "        \"file\": \"" << settings.sceneFile << "\",\n";
    file << "        \"sphere_count\": " << settings.sphereCount << ",\n";
    file << "        \"cylinder_count\": " << settings.cylinderCount << "\n";
    file << "    },\n";

    // Grid section
    file << "    \"grid\": {\n";
    file << "        \"width\": " << settings.gridWidth << ",\n";
    file << "        \"height\": " << settings.gridHeight << ",\n";
    file << "        \"depth\": " << settings.gridDepth << "\n";
    file << "    },\n";

    // Benchmark section
    file << "    \"benchmark\": {\n";
    file << "        \"interleave_width\": " << settings.interleaveWidth << ",\n";
    file << "        \"interleave_height\": " << settings.interleaveHeight << ",\n";
    file << "        \"interleave_angle\": " << settings.interleaveAngle << ",\n";
    file << "        \"interleave_z\": " << settings.interleaveZ << ",\n";
    file << "        \"interleave_y\": " << settings.interleaveY << ",\n";
    file << "        \"radius_factor\": " << settings.radiusFactor << "\n";
    file << "    },\n";

    // Geometry section
    file << "    \"geometry\": {\n";
    file << "        \"separation\": " << settings.separation << ",\n";
    file << "        \"sphere_radius\": " << settings.sphereRadius << ",\n";
    file << "        \"cylinder_radius\": " << settings.cylinderRadius << "\n";
    file << "    },\n";

    // Render section
    file << "    \"render\": {\n";
    file << "        \"screen_width\": " << settings.screenWidth << ",\n";
    file << "        \"screen_height\": " << settings.screenHeight << ",\n";
    file << "        \"with_occlusion_culling\": " << (settings.withOcclusionCulling ? "true" : "false") << ",\n";
    file << "        \"downsample_level\": " << settings.downsampleLevel << ",\n";
    file << "        \"work_group_size_per_pixel_x\": " << settings.workGroupSizePerPixelX << ",\n";
    file << "        \"work_group_size_per_pixel_y\": " << settings.workGroupSizePerPixelY << ",\n";
    file << "        \"work_group_size_per_sphere\": " << settings.workGroupSizePerSphere << ",\n";
    file << "        \"work_group_size_per_cylinder\": " << settings.workGroupSizePerCylinder << "\n";
    file << "    },\n";

    // Profiler section
    file << "    \"profiler\": {\n";
    file << "        \"timer_duration\": " << settings.timerDuration << ",\n";
    file << "        \"benchmarking_enabled\": " << (settings.benchmarkingEnabled ? "true" : "false") << "\n";
    file << "    },\n";

    // Out-of-core section
    file << "    \"outofcore\": {\n";
    file << "        \"atoms_per_block\": " << settings.oocAtomsPerBlock << ",\n";
    file << "        \"max_octree_depth\": " << settings.oocMaxOctreeDepth << ",\n";
    file << "        \"blocks_per_leaf\": " << settings.oocBlocksPerLeaf << ",\n";
    file << "        \"max_block_pool_slots\": " << settings.oocMaxBlockPoolSlots << ",\n";
    file << "        \"max_requests_per_frame\": " << settings.oocMaxRequestsPerFrame << ",\n";
    file << "        \"visibility_threshold\": " << settings.oocVisibilityThreshold << ",\n";
    file << "        \"occlusion_method\": \"" << settings.oocOcclusionMethod << "\",\n";
    file << "        \"stats_accumulate_frames\": " << settings.oocStatsAccumulateFrames << ",\n";
    file << "        \"stats_csv_path\": \"" << settings.oocStatsCsvPath << "\",\n";
    file << "        \"preprocess_stats_csv_path\": \"" << settings.oocPreprocessStatsCsvPath << "\"\n";
    file << "    }\n";

    file << "}\n";

    file.close();
    std::cout << "Configuration saved to: " << configPath << std::endl;
}
