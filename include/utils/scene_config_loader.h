#ifndef SCENE_CONFIG_LOADER_H
#define SCENE_CONFIG_LOADER_H

#include <string>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

/**
 * @struct SceneSettings
 * @brief Structure containing all scene configuration parameters.
 * 
 * This structure holds all the configuration parameters that were previously
 * global variables in benchmark_resources. It allows loading configuration
 * from a JSON file without recompiling the project.
 */
struct SceneSettings
{
    // Scene type and file
    std::string sceneType = "LOADED_SCENE";  // "LOADED_SCENE", "GRID_SCENE", "PACKAGE_SCENE"
    std::string sceneFile = "1AGA.mmtf";
    int sphereCount = 552008;
    int cylinderCount = 519767;

    // Grid settings
    int gridWidth = 10;
    int gridHeight = 10;
    int gridDepth = 10;

    // Benchmark settings
    int interleaveWidth = 5;
    int interleaveHeight = 5;
    int interleaveAngle = 8;
    int interleaveZ = 5;
    int interleaveY = 5;
    float radiusFactor = 2.0f;

    // Geometry settings
    float separation = 2.0f;
    float sphereRadius = 0.5f;
    float cylinderRadius = 0.2f;

    // Render settings
    int screenWidth = 1024;
    int screenHeight = 1024;
    bool withOcclusionCulling = false;
    int downsampleLevel = 4;
    int workGroupSizePerPixelX = 16;
    int workGroupSizePerPixelY = 16;
    int workGroupSizePerSphere = 256;
    int workGroupSizePerCylinder = 256;

    // Profiler settings
    float timerDuration = 48.0f;
    bool benchmarkingEnabled = false;

    // Out-of-core settings
    int oocAtomsPerBlock       = 512;
    int oocMaxOctreeDepth      = 10;
    int oocBlocksPerLeaf       = 4;
    int oocMaxBlockPoolSlots   = 2048;
    int oocMaxRequestsPerFrame = 256;
    float oocVisibilityThreshold = 0.01f;
    // "none", "probabilistic", "probabilistic_overlap", "hiz", "hiz_probabilistic"
    std::string oocOcclusionMethod = "none";

    /** If > 0, accumulate OOC metrics over this many frames, then append one CSV row. */
    int oocStatsAccumulateFrames = 0;
    /** Output path for batch statistics (append mode). */
    std::string oocStatsCsvPath = "media/csv/cuda_outofcore/ooc_stats_batch.csv";

    /** If non-empty, append one CSV row with preprocess timings, host sizes, and estimated VRAM after init. */
    std::string oocPreprocessStatsCsvPath = "media/csv/cuda_outofcore/ooc_preprocess_metrics.csv";

    /**
     * @brief Gets the full path to the scene file.
     * @return Filesystem path to the scene file.
     */
    std::filesystem::path getScenePath() const 
    {
        return std::filesystem::path("assets/molecules") / sceneFile;
    }
};

/**
 * @class SceneConfigLoader
 * @brief Loads scene configuration from a JSON file.
 * 
 * This class provides functionality to load scene configuration from a JSON file,
 * eliminating the need to recompile the project when changing scene parameters.
 */
class SceneConfigLoader
{
public:
    /**
     * @brief Default constructor.
     */
    SceneConfigLoader() = default;

    /**
     * @brief Loads configuration from a JSON file.
     * 
     * @param configPath Path to the JSON configuration file.
     * @return SceneSettings structure with loaded values.
     * @throws std::runtime_error if the file cannot be read or parsed.
     */
    static SceneSettings loadFromFile(const std::string& configPath);

    /**
     * @brief Loads configuration from the default config file.
     * 
     * Uses "assets/config/scene_config.json" as the default path.
     * 
     * @return SceneSettings structure with loaded values.
     */
    static SceneSettings loadDefault();

    /**
     * @brief Applies loaded settings to the global variables in benchmark_resources.
     * 
     * This function updates the global variables used by the existing code
     * to maintain backwards compatibility.
     * 
     * @param settings The settings to apply.
     */
    static void applyToGlobals(const SceneSettings& settings);

    /**
     * @brief Saves current settings to a JSON file.
     * 
     * @param settings The settings to save.
     * @param configPath Path to the output JSON file.
     */
    static void saveToFile(const SceneSettings& settings, const std::string& configPath);

private:
    /**
     * @brief Parses a JSON string value.
     */
    static std::string parseString(const std::string& json, const std::string& key, const std::string& defaultValue);

    /**
     * @brief Parses a JSON integer value.
     */
    static int parseInt(const std::string& json, const std::string& key, int defaultValue);

    /**
     * @brief Parses a JSON float value.
     */
    static float parseFloat(const std::string& json, const std::string& key, float defaultValue);

    /**
     * @brief Parses a JSON boolean value.
     */
    static bool parseBool(const std::string& json, const std::string& key, bool defaultValue);
};

#endif // SCENE_CONFIG_LOADER_H
