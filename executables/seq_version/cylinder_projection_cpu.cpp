#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glad/glad.h>
#include <SDL3/SDL.h>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <string>

#include "appSDLRenderer.h"

#include "algorithms/frustum_cull.h"

#include "utils/benchmark_resources.h"
#include "utils/sequential/aux_functions_cylinder.h"

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
int SCR_WIDTH = 1024;
int SCR_HEIGHT = 1024;

int sphere_count = 1024*64;
int frustumSpheres = 0;
int visibleSpheres = 0;

std::string title = "Brute Sequential Method"; 

bool shown = true;

int topLevel = 4;
int level = 3;
bool showHiz = false;

struct Cylinder
{
    glm::vec3 pa; // extreme A
    glm::vec3 pb; // extreme B
    float radius;
};

void renderFrame(std::vector<uint32_t>& framebuffer, 
    std::vector<float>& depthBuffer, std::vector<Cylinder>& cylinders,
    Camera& cam) 
{
    const glm::mat4& proj = cam.getProjection();
    const glm::mat4& view = cam.getView();
    const glm::vec3& up = cam.getUp();
    const glm::vec3& right = cam.getRight();
    const glm::vec3& front = cam.getFront();
    const glm::vec3& camPos = cam.getPosition();
    const float& fov = cam.getFov();
    const Frustum frustum(cam);
    
    for(int i = 0; i < cylinders.size(); i++)
    {
       drawCylinder(proj, view, up, front, right, camPos, SCR_WIDTH, SCR_HEIGHT, 
            glm::vec4(cylinders[i].pa, 1.0f), glm::vec4(cylinders[i].pb, 1.0f), cylinders[i].radius, 
            fov, framebuffer, depthBuffer);
    }
    
}

int main(int argc, char* argv[]) 
{
    std::unordered_map<std::string, int*> scene_data = {
        {"Screen width", &SCR_WIDTH},
        {"Screen height", &SCR_HEIGHT},
        {"Sphere count", &sphere_count},
        {"Spheres On Frustum", &frustumSpheres},
        {"Visible Spheres", &visibleSpheres}
    };

    AppRenderer window { title, SCR_WIDTH, SCR_HEIGHT, shown };
    Camera camera(SCR_WIDTH, SCR_HEIGHT);
    camera.SetPosition(.0f, .0f, .0f);
    CameraController camera_controller(window, camera);
    
    std::vector<uint32_t> framebuffer(SCR_WIDTH * SCR_HEIGHT);
    std::vector<float> depthBuffer(SCR_WIDTH * SCR_HEIGHT, FLT_MAX);
    
   
    std::vector<Cylinder> cylinders = {
        {{0.0f, 0.0f, 0.0f}, {4.0f, 0.0f, 0.0f}, 1.0f},
        {{0.0f, 2.0f, 0.0f}, {4.0f, 2.0f, 0.0f}, 1.0f},
        {{0.0f, 4.0f, 0.0f}, {4.0f, 4.0f, 0.0f}, 1.0f},
    
    };
    
    window.setupSceneInfoGui("Scene Info", scene_data);
    window.setupCameraGui("Camera Info", &camera);
    window.setupInputInfoGui("General Input Info");

    std::fill(depthBuffer.begin(), depthBuffer.end(), FLT_MAX);
    try
    {
        bool isRunning = true;
        while ( isRunning )
        {
            
            camera_controller.cameraUpdate();


            std::fill(framebuffer.begin(), framebuffer.end(), 0xFFFFFF00);
            
            renderFrame(framebuffer, depthBuffer, cylinders, camera);

            std::fill(depthBuffer.begin(), depthBuffer.end(), FLT_MAX);

            window.updateTexture(framebuffer);

            isRunning = window.update();


            if (window.getInput().isKeyDown(Key::H)) 
            {
                std::cout << "Show hiz " << showHiz << std::endl;
                showHiz = !showHiz;
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




