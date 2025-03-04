#ifndef _CAMERA_CONTROLLER_H
#define _CAMERA_CONTROLLER_H

#include "ux/camera.h"
#include "vis/window.h"

class CameraController
{
    public:
        CameraController(Window& window, Camera& camera);
        ~CameraController() = default;

        void mouseAction() const;
        void keyBoardAction() const;
        void cameraUpdate();

        void setBenchmark(bool benchmark);
        bool onBenchmark();
    
    private:
        Window* m_window;
        Camera* m_camera;
        const Input* m_input;
        bool m_hasBenchmark = false;
        bool m_onBenchmark = false;


};

#endif // _CAMERA_CONTROLLER_H