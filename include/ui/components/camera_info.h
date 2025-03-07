#ifndef _CAMERA_INFO_H_
#define _CAMERA_INFO_H_

#include <vector>
#include "ux/camera.h"
#include "ui/component.h"

class CameraInfoComponent : public Component
{
public:
    CameraInfoComponent(std::string name, Camera* camera);

public:
    Camera* m_camera;
};


#endif // _CAMERA_INFO_H_