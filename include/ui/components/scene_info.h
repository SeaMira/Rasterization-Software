#ifndef _SCENE_INFO_H_
#define _SCENE_INFO_H_

#include <unordered_map>
#include "ui/component.h"
#include "ui/element.h"

/**
 * @class SceneInfoComponent
 * @brief Represents a UI component for displaying scene information.
 * 
 * This class provides functionality for rendering a UI component that displays information about the current scene, 
 * such as screen width, screen height, sphere count, spheres on frustum and visible spheres.
 */
class SceneInfoComponent : public Component
{
public:
    /**
     * @brief Constructs a SceneInfoComponent with the specified name and scene data.
     * 
     * This will create the interface elements necessary to display the scene information looping
     * through the map keys and values.
     * 
     * @param name The name of the component.
     * @param scene_data The scene data which contains labels and values in the UI as keys and values in the map.
     */
    SceneInfoComponent(std::string name, std::unordered_map<std::string, int*> scene_data);

};


#endif // _SCENE_INFO_H_