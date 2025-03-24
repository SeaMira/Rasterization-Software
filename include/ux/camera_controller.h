#ifndef _CAMERA_CONTROLLER_H
#define _CAMERA_CONTROLLER_H

#include "ux/camera.h"
#include "vis/window.h"

/**
 * @class CameraController
 * 
 * @brief Class for controlling the camera at runtime.
 * 
 * The CameraController class is used to control the camera at runtime, it allows the camera to move 
 * by capturing the input from the window and updating the camera accordingly.
 */
class CameraController
{
    public:
        /**
         * @brief Constructor for the CameraController class.
         * 
         * Constructor for the CameraController class, initializes the camera controller with the window and the camera. Manages its
         * own input object.
         */
        CameraController(Window& window, Camera& camera);

        /**
         * @brief destructorof the CameraController class.
         * 
         * Default constructor, no need for manual deallocation.
         */
        ~CameraController() = default;

        /**
         * @brief updates the camera setting on mouse input.
         */
        void mouseAction() const;

        /**
         * @brief updates the camera setting on keyboard input.
         */
        void keyBoardAction() const;

        /**
         * @brief updates the camera settings.
         * 
         * Updates the camera settings by checking the input and updating the camera accordingly. The movement can
         * be from the keyboard or the mouse, or from benchmark.
         */
        void cameraUpdate();

        /**
         * @brief sets the benchmark flag.
         * 
         * Sets the benchmark flag so the camera wil update through the benchmark.
         * 
         * @param benchmark the flag to set the benchmark.
         */
        void setBenchmark(bool benchmark);

        /**
         * @brief gets the benchmark flag.
         * 
         * Gets the benchmark flag so the camera will update through the benchmark or not.
         * 
         * @return the benchmark flag.
         */
        bool onBenchmark();

        /**
         * @brief gets the on-benchmark location data.
         * 
         * Gets the on-benchmark location data so it can be accessed from different objects.
         * 
         * @return on-benchmark bool location data.
         */
        bool* onBenchmarkData() { return &m_onBenchmark; };

        /**
         * @brief gets the input handle.
         * 
         * Gets the input handle so it can be accessed from different objects. Has info on
         * keys/button pressed.
         * 
         * @return the input handle.
         */
        inline const Input* inputHandle() { return m_input; }

        /**
         * @brief gets the camera handle.
         * 
         * Gets the camera handle so it can be accessed from different objects. Has info on
         * camera settings.
         */
        inline Camera* cameraHandle() { return m_camera; }
    
    private:
        Window* m_window; ///< the window to be used for the camera controller.
        Camera* m_camera; ///< the camera to be controlled.
        const Input* m_input; ///< the input to analyze keyboard and mouse input.
        bool m_hasBenchmark = false; ///< flag for having benchmark.
        bool m_onBenchmark = false; ///< flag for being on benchmark.


};

#endif // _CAMERA_CONTROLLER_H