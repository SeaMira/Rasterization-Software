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
#include "utils/sequential/aux_functions_sphere.h"

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

int sphere_count = 1024;
int frustumSpheres = 0;
int visibleSpheres = 0;

std::string title = "Brute Sequential Method"; 

bool shown = true;

bool withOcclusionCulling = true;

int topLevel = 4;
int level = 3;
bool showHiz = false;

HierarchicalZBuffer hizPyramid;
std::vector<uint8_t> spheresVisibilityLastFrameCache(sphere_count, 0);
std::vector<uint8_t> spheresVisibilityFrameCache(sphere_count, 10);
std::vector<int> pixelOwnership(SCR_WIDTH*SCR_HEIGHT, -1);

std::unordered_map<int, int> buildSpherePixelCount(const std::vector<int>& pixelOwnership) 
{
    std::unordered_map<int, int> spherePixelCount;

    for (int i = 0; i < pixelOwnership.size(); i++) 
        if (pixelOwnership[i] != -1)
            spherePixelCount[pixelOwnership[i]]++;

    return spherePixelCount;
}

std::unordered_map<int, int> ownedPixelsMap = buildSpherePixelCount(pixelOwnership);

void renderFrameWithOcclusionCulling(std::vector<uint32_t>& framebuffer, 
    std::vector<float>& depthBuffer, std::vector<glm::vec4>& spheres,
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
    int frustumSpheresCount = 0;
    int visibleSpheresCount = 0;
    for(int i = 0; i < spheres.size(); i++)
    {
        // std::cout << static_cast<int>(spheresVisibilityFrameCache[i]) << std::endl;
        if (frustum.isSphereInside(spheres[i]))
        {
            bool wasDrawn = drawSphereWithOcclusionCulling(proj, view, up, front, right, camPos, SCR_WIDTH, SCR_HEIGHT, fov, spheres[i], 
                framebuffer, depthBuffer, hizPyramid, spheresVisibilityFrameCache[i], pixelOwnership, i, ownedPixelsMap[i]);
            frustumSpheresCount++;
            if (wasDrawn)
            {
                visibleSpheresCount++;
                spheresVisibilityFrameCache[i] = spheresVisibilityFrameCache[i] | 0b10000000;
            } 
        } else spheresVisibilityFrameCache[i] = 0;
    }
    ownedPixelsMap = buildSpherePixelCount(pixelOwnership);
    for(int i = 0; i < spheres.size(); i++)
    {
        if (spheresVisibilityFrameCache[i] & 0b10000000)
        {
            if (spheresVisibilityLastFrameCache[i] <= (spheresVisibilityFrameCache[i] & 0b01111111) &&
                ownedPixelsMap[i] > 0) 
                spheresVisibilityFrameCache[i] = 10;
        }
        spheresVisibilityLastFrameCache[i] = spheresVisibilityFrameCache[i] & 0b01111111;
    }
    std::fill(pixelOwnership.begin(), pixelOwnership.end(), -1);

    frustumSpheres = frustumSpheresCount;
    visibleSpheres = visibleSpheresCount;
    
}

void renderFrameWithoutOcclusionCulling(std::vector<uint32_t>& framebuffer, 
    std::vector<float>& depthBuffer, std::vector<glm::vec4>& spheres,
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
    int frustumSpheresCount = 0;
    int visibleSpheresCount = 0;
    frustumSpheres = 0;
    for(int i = 0; i < spheres.size(); i++)
    {
        if (frustum.isSphereInside(spheres[i]))
        {
            frustumSpheres++;
            drawSphere(proj, view, up, front, right, camPos, SCR_WIDTH, SCR_HEIGHT, fov, spheres[i], framebuffer, depthBuffer);
        }
    }
    visibleSpheres = frustumSpheres;
    
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
    
   
    std::vector<glm::vec4> spheres = getScene(sphere_count);
    std::vector<std::pair<glm::vec3, glm::vec3>> chkPoints = getCheckpoints(sphere_count, spheres);
    
    Benchmark benchmark(camera_controller, chkPoints);
    Profiler profiler(window, (withOcclusionCulling ? "media/off/seq/occ_frame_times.off": "media/off/seq/frame_times.off"), 
        (withOcclusionCulling ? "media/off/seq/occ_process_times.off": "media/off/seq/process_times.off"), sphere_count, 0);
    
    window.setupSceneInfoGui("Scene Info", scene_data);
    window.setupCameraGui("Camera Info", &camera);
    window.setupInputInfoGui("General Input Info");
    window.setupBenchmarkInfoGui("Benchmark", &benchmark);

    std::fill(depthBuffer.begin(), depthBuffer.end(), FLT_MAX);
    hizPyramid = generateHiZPyramid(DepthBuffer{SCR_WIDTH, SCR_HEIGHT, camera.getFar(), depthBuffer}, topLevel);
    try
    {
        bool isRunning = true;
        while ( isRunning )
        {
            
            camera_controller.cameraUpdate();
            benchmark.update();

            profiler.updateProfiler(benchmark.getCheckpointID(), frustumSpheres, visibleSpheres, 0, 0);
            if (window.getInput().isKeyDown(Key::T)) profiler.startSavingNextFrames(benchmark.getCheckpointID());

            std::fill(framebuffer.begin(), framebuffer.end(), 0xFFFFFF00);
            if (withOcclusionCulling)
            {
                renderFrameWithOcclusionCulling(framebuffer, depthBuffer, spheres, camera);
                hizPyramid = generateHiZPyramid(DepthBuffer{SCR_WIDTH, SCR_HEIGHT, camera.getFar(), depthBuffer}, topLevel);
            } else

            if (showHiz) drawMipmaps(hizPyramid, level, framebuffer, SCR_WIDTH, SCR_HEIGHT);
            std::fill(depthBuffer.begin(), depthBuffer.end(), FLT_MAX);

            window.updateTexture(framebuffer);

            isRunning = window.update();

            if (window.getInput().isKeyDown(Key::F10)) 
            {
                std::string sshot_name  = "media/img/seq/frame_" + std::to_string(benchmark.getCheckpointID()) + ".bmp";
                takeScreenshot(window.getRenderer(), sshot_name);   
            }

            if (window.getInput().isKeyDown(Key::O)) 
            {
                withOcclusionCulling = !withOcclusionCulling;
                std::cout << "Occlusion Culling: " << (withOcclusionCulling ? "On" : "Off") << std::endl;
            }
            
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




