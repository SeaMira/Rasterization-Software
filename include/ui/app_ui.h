#ifndef _APP_UI_H_
#define _APP_UI_H_

#include <imgui.h>
#include <memory>
#include <unordered_map>
#include <string>
#include "ui/component.h"
#include "ui/components/scene_info.h"
#include "ui/components/camera_info.h"
#include "ui/components/input_info.h"
#include "ui/components/benchmark_info.h"
#include "ui/components/pipeline_config.h"

/**
 * @class AppUI
 * @brief Represents a UI app that may contain many components as submodules.
 * 
 * This class allows for the creation of a UI app that can present multiple collapsable components, making it flexible
 * and modular. It provides functionality for adding components and rendering them in the ImGui interface.
 */
class AppUI
{
public:
    /**
     * @brief Adds a SceneInfoComponent to the UI app.
     * @param name The name of the component.
     * @param scene_data A map of scene data to be displayed in the component.
     */
    void addSceneInfoComponent(std::string name, std::unordered_map<std::string, int*> scene_data);
    
    /**
     * @brief Adds a CameraInfoComponent to the UI app.
     * @param name The name of the component.
     * @param camera A pointer to the camera object whose data will be displayed in the component.
     */
    void addCameraInfoComponent(std::string name, Camera* camera);

    /**
     * @brief Adds an InputInfoComponent to the UI app.
     * 
     * @param name The name of the component.
     * @param input A pointer to the input object whose data will be displayed in the component.
     */
    void addInputInfoComponent(std::string name, Input* input);

    /**
     * @brief Adds a BenchmarkInfoComponent to the UI app.
     * 
     * @param name The name of the component.
     * @param benchmark A pointer to the benchmark object whose data will be displayed in the component.
     */
    void addBenchmarkInfoComponent(std::string name, Benchmark* benchmark);

    /**
     * @brief Adds a PipelineConfigComponent to the UI app.
     * 
     * @param name The name of the component.
     * @param smallThreshold Reference to the small entity threshold value.
     * @param tileSize Reference to the tile size value.
     * @param maxEntitiesPerTile Reference to max entities per tile.
     * @param tilesX Reference to number of tiles in X.
     * @param tilesY Reference to number of tiles in Y.
     */
    void addPipelineConfigComponent(std::string name, 
                                    int& smallThreshold,
                                    int& tileSize,
                                    int& maxEntitiesPerTile,
                                    int& tilesX,
                                    int& tilesY);

    /**
     * @brief Renders the UI app by rendering all its associated components.
     * 
     * This method loops through all the components contained within the app and renders them in a collapsable submenu each.
     */
    void render();
    
private:
    std::vector<std::unique_ptr<Component>> components; ///< The list of components associated with the UI app.
};

#endif