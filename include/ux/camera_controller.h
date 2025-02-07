#ifndef _CAMERA_CONTROLLER_H
#define _CAMERA_CONTROLLER_H

#include "ux/camera.h"
#include "vis/window.h"

class CameraController
{
    public:
        CameraController(Window& window, Camera& camera);
        ~CameraController() = default;

        void keyBoardAction() const;
        void mouseAction() const;
        void cameraUpdate() const;
    
    private:
        Window* m_window;
        Camera* m_camera;
        const Input* m_input;

};

#endif // _CAMERA_CONTROLLER_H