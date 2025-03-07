#include "ui/app_ui.h"

void AppUI::addSceneInfoComponent(std::string name, std::unordered_map<std::string, int*> scene_data)
{
    components.push_back(std::make_unique<SceneInfoComponent>(name, scene_data));
}

void AppUI::addCameraInfoComponent(std::string name, Camera* camera)
{
    components.push_back(std::make_unique<CameraInfoComponent>(name, camera));
}

void AppUI::render()
{
    for (auto& component : components)
    {
        component->render();
    }
}