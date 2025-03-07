#include "ui/components/camera_info.h"

CameraInfoComponent::CameraInfoComponent(std::string name, Camera* camera) : 
    Component(name), m_camera(camera) 
{
    addElement(std::make_unique<Vec3TextElement>("position", camera->cameraPos));
    addElement(std::make_unique<Vec3TextElement>("up", camera->cameraUp));
    addElement(std::make_unique<Vec3TextElement>("Front", camera->cameraFront));
    
    // Speed and FOV
    addElement(std::make_unique<TextElementf>("Speed", camera->mSpeed));
    ImGui::SameLine();
    addElement(std::make_unique<TextElementf>("FOV (°)", camera->fov));
    
    // Yaw and Pitch
    addElement(std::make_unique<TextElementf>("Yaw (°)", camera->yaw));
    ImGui::SameLine();
    addElement(std::make_unique<TextElementf>("Pitch (°)", camera->pitch));

}

