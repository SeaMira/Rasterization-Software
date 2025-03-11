#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glad/glad.h>
#include <SDL3/SDL.h>
#include <vector>
#include <cmath>
#include <string>

#include "appSDLRenderer.h"

#include "utils/benchmark_resources.h"
#include "utils/sequential/aux_functions.h"

#include "ux/input.h"
#include "ux/camera_controller.h"
#include "ux/cinematic/benchmark.h"
#include "ux/profiler/profiler.h"

#include "vis/window.h"
#include "vis/gl/frame_buffer.h"
#include "vis/gl/texture.h"
#include "vis/canvas.h"
#include "vis/compute_shader_program.h"


// Settings
int SCR_WIDTH = 800;
int SCR_HEIGHT = 600;

int sphere_count = 126;

std::string title = "Brute Sequential Method"; 

bool shown = true;

void renderFrame(std::vector<uint32_t>& framebuffer, 
    std::vector<float>& depthBuffer, std::vector<glm::vec4>& spheres,
    Camera& cam) 
{
    const glm::mat4& proj = cam.getProjection();
    const glm::mat4& view = cam.getView();
    const glm::vec3& up = cam.getUp();
    // const glm::vec3& right = cam.getRight();
    const glm::vec3& front = cam.getFront();
    const glm::vec3& camPos = cam.getPosition();
    // const glm::vec2 resol = glm::vec2(SCR_WIDTH, SCR_HEIGHT);
    
    for (const auto& sphere : spheres) 
        drawSphere(proj, view, up, front, camPos, SCR_WIDTH, SCR_HEIGHT, sphere, 
            framebuffer, depthBuffer);
    
}

int main(int argc, char* argv[]) 
{
    std::unordered_map<std::string, int*> scene_data = {
        {"Screen width", &SCR_WIDTH},
        {"Screen height", &SCR_HEIGHT},
        {"Sphere count", &sphere_count}
    };

    AppRenderer window { title, SCR_WIDTH, SCR_HEIGHT, shown };
    Camera camera(SCR_WIDTH, SCR_HEIGHT);
    camera.SetPosition(.0f, .0f, .0f);
    CameraController camera_controller(window, camera);
    
    std::vector<uint32_t> framebuffer(SCR_WIDTH * SCR_HEIGHT);
    std::vector<float> depthBuffer(SCR_WIDTH * SCR_HEIGHT, FLT_MAX);
    
   
    std::vector<glm::vec4> spheres = getScene(sphere_count);
    std::vector<std::pair<glm::vec3, glm::vec3>> chkPoints = getCheckpoints(sphere_count, spheres);
    
    Benchmark benchmark(camera_controller, chkPoints);
    Profiler profiler(window, "media/off/seq/frame_times.off", "media/off/seq/process_times.off");
    
    window.setupSceneInfoGui("Scene Info", scene_data);
    window.setupCameraGui("Camera Info", &camera);
    window.setupInputInfoGui("General Input Info");
    window.setupBenchmarkInfoGui("Benchmark", &benchmark);
    try
    {
        bool isRunning = true;
        while ( isRunning )
        {
            
            camera_controller.cameraUpdate();
            benchmark.update();

            profiler.updateProfiler();
            if (window.getInput().isKeyDown(Key::T)) profiler.startSavingNextFrames(benchmark.getCheckpointID());

            std::fill(framebuffer.begin(), framebuffer.end(), 0xFFFFFF00);
            std::fill(depthBuffer.begin(), depthBuffer.end(), FLT_MAX);
            
            renderFrame(framebuffer, depthBuffer, spheres, camera);
            
            window.updateTexture(framebuffer);

            isRunning = window.update();

            if (window.getInput().isKeyDown(Key::F10)) 
            {
                std::string sshot_name  = "media/img/seq/frame_" + std::to_string(benchmark.getCheckpointID()) + ".bmp";
                takeScreenshot(window.getRenderer(), sshot_name);   
            }

        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}




