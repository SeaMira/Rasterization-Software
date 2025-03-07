#include "ui/components/scene_info.h"

SceneInfoComponent::SceneInfoComponent(std::string name, std::unordered_map<std::string, int*> scene_data) : 
    Component(name) 
{
    for (const auto& [key, value] : scene_data) 
    {
        addElement(std::make_unique<TextElementi>(key, value));
    }
}