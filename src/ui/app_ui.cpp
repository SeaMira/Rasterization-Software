#include "ui/app_ui.h"

void AppUI::addSceneInfoComponent(std::string name, std::unordered_map<std::string, int*> scene_data)
{
    components.push_back(std::make_unique<SceneInfoComponent>(name, scene_data));
}

void AppUI::addCameraInfoComponent(std::string name, Camera* camera)
{
    components.push_back(std::make_unique<CameraInfoComponent>(name, camera));
}

void AppUI::addInputInfoComponent(std::string name, Input* input)
{
    components.push_back(std::make_unique<InputInfoComponent>(name, input));
}

void AppUI::addBenchmarkInfoComponent(std::string name, Benchmark* benchmark)
{
    components.push_back(std::make_unique<BenchmarkInfoComponent>(name, benchmark));
}

void AppUI::addPipelineConfigComponent(std::string name, 
                                       int& smallThreshold,
                                       int& tileSize,
                                       int& maxEntitiesPerTile,
                                       int& tilesX,
                                       int& tilesY)
{
    components.push_back(std::make_unique<PipelineConfigComponent>(
        name, smallThreshold, tileSize, maxEntitiesPerTile, tilesX, tilesY));
}

void AppUI::addOocTuningComponent(std::string name, float& visibilityThreshold)
{
    components.push_back(std::make_unique<OocTuningComponent>(std::move(name), visibilityThreshold));
}

void AppUI::render()
{
    for (auto& component : components)
    {
        if (ImGui::CollapsingHeader(component->getName().c_str())) 
            component->render();
    }
}