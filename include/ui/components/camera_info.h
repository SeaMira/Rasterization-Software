#ifndef _CAMERA_INFO_H_
#define _CAMERA_INFO_H_

#include <vector>
#include "ux/camera.h"
#include "ui/component.h"

/**
 * @class CameraInfoComponent
 * @brief Represents a UI component for displaying camera information.
 * 
 * This class provides functionality for rendering a UI component that displays information about the current camera: 
 * position, front, up, right, yaw, pitch, speed, FOV, near and far, letting the user interact with some of the values.
 */
class CameraInfoComponent : public Component
{
public:
    /**
     * @brief Constructs a CameraInfoComponent with the specified name and camera object.
     * 
     * This will create a the interface elements necessary to display and interact with the camera information.
     * 
     * @param name The name of the component.
     * @param camera The camera object to display information from.
     */
    CameraInfoComponent(std::string name, Camera* camera);

private:
    Camera* m_camera; ///< The camera object associated with the component.
};


#endif // _CAMERA_INFO_H_