#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glad/glad.h>
#include <SDL3/SDL.h>

#include "ux/input.h"
#include "vis/window.h"
#include "ux/camera_controller.h"

#include "vis/gl/frame_buffer.h"
#include "vis/gl/texture.h"
#include "vis/canvas.h"
#include "vis/compute_shader_program.h"

// Settings
const int SCR_WIDTH = 800;
const int SCR_HEIGHT = 600;


std::string title = "Input testing";

bool shown = true;
bool show_grid = true;
bool show_axes = true;
bool show_num = true;
bool show_fps = true;

int main(int argc, char* argv[]) {
    Window window { title, SCR_WIDTH, SCR_HEIGHT, shown };
    Camera camera(SCR_WIDTH, SCR_HEIGHT);
    camera.SetPosition(3.0f, 0.0f, 0.0f);
    CameraController camera_controller(window, camera);
    ComputeShader computeShader("raster.compute");
    computeShader.use();
    
    Canvas canvas(GL_TEXTURE_2D, GL_RGBA8, SCR_WIDTH, SCR_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE);
    canvas.setFBO(GL_COLOR_ATTACHMENT0);
    
    if (!canvas.getFramebuffer().isComplete()) {
        throw std::runtime_error("Error: Framebuffer incompleto.");
    }
    
    canvas.bindTexture(0);
    canvas.bindFBO();

    try
    {
        bool isRunning = true;
        SDL_Event event;
        while ( isRunning )
        {
            if (window.getInput().isKeyDown(Key::G)) show_grid = !show_grid;
            if (window.getInput().isKeyDown(Key::H)) show_axes = !show_axes;
            if (window.getInput().isKeyDown(Key::J)) show_num = !show_num;
            if (window.getInput().isKeyDown(Key::F)) show_fps = !show_fps;
            

            if (show_fps) std::cout << 1/window.getInput().deltaTime << std::endl;
            computeShader.use();
            computeShader.setMat4("viewMatrix", camera.getView());
            computeShader.setVec3("front", camera.getFront());
            computeShader.setVec3("up", camera.getUp());
            computeShader.setVec3("right", camera.getRight());
            computeShader.setVec3("cameraPos", camera.getPosition());
            computeShader.setVec2("screenResolution", SCR_WIDTH, SCR_HEIGHT);
            computeShader.setFloat("iTime", (float)window.getElapsedTime());
            computeShader.setFloat("FOV", camera.getFov());
            computeShader.setFloat("show_grid", show_grid);
            computeShader.setFloat("show_axis", show_axes);
            computeShader.setFloat("show_num", show_num);

            const Input & inputs = window.getInput();
            if ( inputs.windowResized )
            {
                camera.SetScrSize(inputs.windowSize.x, inputs.windowSize.y);
            }

            camera_controller.keyBoardAction();
            camera_controller.mouseAction();

            glDispatchCompute((SCR_WIDTH + 15) / 16, (SCR_HEIGHT + 15) / 16, 1);
            glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
            // Blit del framebuffer al default framebuffer (pantalla)
            glBindFramebuffer(GL_READ_FRAMEBUFFER, canvas.getFramebuffer().getId());
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
            glBlitFramebuffer(0, 0, SCR_WIDTH, SCR_HEIGHT, 0, 0, SCR_WIDTH, SCR_HEIGHT, GL_COLOR_BUFFER_BIT, GL_NEAREST);

            isRunning = window.update();
        }
    }
    catch ( const std::exception & e )
    {
        throw std::runtime_error(e.what());
    }

    return 0;
}