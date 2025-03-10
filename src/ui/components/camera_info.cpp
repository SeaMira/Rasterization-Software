#include "ui/components/camera_info.h"

CameraInfoComponent::CameraInfoComponent(std::string name, Camera* camera) : 
    Component(name), m_camera(camera) 
{
    addElement(std::make_unique<Vec3TextElement>("Position", camera->cameraPos));
    addElement(std::make_unique<Vec3TextElement>("Up", camera->cameraUp));
    addElement(std::make_unique<Vec3TextElement>("Front", camera->cameraFront));
    
    // Yaw and Pitch
    addElement(std::make_unique<TextElementf>("Yaw (°)", camera->yaw));
    addElement(std::make_unique<SameLineElement>());
    addElement(std::make_unique<TextElementf>("Pitch (°)", camera->pitch));
    
    // Speed and FOV
    addElement(std::make_unique<SliderElement>("Speed", camera->mSpeed, 1.0f, 30.0f));
    addElement(std::make_unique<SliderElement>("FOV (°)", camera->fov, 1.0f, 180.0f));
    
    // Near and Far
    addElement(std::make_unique<SliderElementTopBounded>("Near", camera->mNear, 0.1f, camera->mFar));
    addElement(std::make_unique<SliderElementLowBounded>("Far", camera->mFar, camera->mNear, 1000.0f));
    

}

