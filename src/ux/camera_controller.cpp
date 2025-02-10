#include <iostream>
#include "ux/camera_controller.h"

CameraController::CameraController(Window& window, Camera& camera) :
    m_window(&window), m_camera(&camera) 
{
    m_input = &(m_window->getInput());
}

void CameraController::keyBoardAction() const
{
    if (m_input->isKeyPressed(Key::W)) 
    {
        m_camera->OnKeyboard('w', m_input->deltaTime);  // Movimiento hacia adelante
    }
    if (m_input->isKeyPressed(Key::S)) 
    {
        m_camera->OnKeyboard('s', m_input->deltaTime);  // Movimiento hacia atrás
    }
    if (m_input->isKeyPressed(Key::A)) 
    {
        m_camera->OnKeyboard('a', m_input->deltaTime);  // Movimiento hacia la izquierda
    }
    if (m_input->isKeyPressed(Key::D)) 
    {
        m_camera->OnKeyboard('d', m_input->deltaTime);  // Movimiento hacia la derecha
    }
    if (m_input->isKeyPressed(Key::E)) 
    {
        m_camera->OnKeyboard('e', m_input->deltaTime);  
    }
    if (m_input->isKeyPressed(Key::Q)) 
    {
        m_camera->OnKeyboard('q', m_input->deltaTime);  
    }
    if (m_input->isKeyPressed(Key::Space)) 
    {
        m_camera->OnKeyboard(' ', m_input->deltaTime);  
    }
    if (m_input->isKeyPressed(Key::LShift)) 
    {
        m_camera->OnKeyboard('l', m_input->deltaTime);  
    }
}

void CameraController::mouseAction() const
{
    m_camera->OnMouse(m_input->mousePosition.x, m_input->mousePosition.y);
    m_camera->OnScroll(static_cast<float>(m_input->deltaMouseWheel));
    m_camera->OnRender(m_input->deltaTime);
}

void CameraController::cameraUpdate() const
{

}