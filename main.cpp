#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glad/glad.h>
#include <SDL3/SDL.h>

#include "ux/input.h"
#include "vis/window.h"
#include "ux/camera_controller.h"

// Settings
const int SCR_WIDTH = 800;
const int SCR_HEIGHT = 600;
std::string title = "Input testing";
bool shown = true;

int main(int argc, char* argv[]) {
    Window window { title, SCR_WIDTH, SCR_HEIGHT, shown };
    Camera camera(SCR_WIDTH, SCR_HEIGHT);
    CameraController camera_controller(window, camera);

    try
    {
        bool isRunning = true;
        while ( isRunning )
        {
            const Input & inputs = window.getInput();
            if ( inputs.windowResized )
            {
                camera.SetScrSize(inputs.windowSize.x, inputs.windowSize.y);
            }

            camera_controller.keyBoardAction();
            camera_controller.mouseAction();
            isRunning = window.update();
        }
    }
    catch ( const std::exception & e )
    {
        throw std::runtime_error(e.what());
    }
}