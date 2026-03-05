#include <iostream>
#include <fstream>

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
#include "utils/scene_config_loader.h"
#include "utils/sequential/aux_functions_cylinder.h"
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

int sphere_count = 4000000;
int cylinder_count = 4000000;
int frustumSpheres = 0;
int visibleSpheres = 0;

int frustumCylinders = 0;
int visibleCylinders = 0;

std::string title = "Sequential Method: Spheres and Cylinders - CPU Frustum Culling"; 

bool shown = true;
bool withOcclusionCulling = false;

int topLevel = 3;
int level = 3;
bool showHiz = false;

HierarchicalZBuffer hizPyramid;

std::vector<uint8_t> spheresVisibilityLastFrameCache(sphere_count, 0);
std::vector<uint8_t> spheresVisibilityFrameCache(sphere_count, 10);

std::vector<uint8_t> cylinderVisibilityLastFrameCache(cylinder_count, 0);
std::vector<uint8_t> cylinderVisibilityFrameCache(cylinder_count, 10);

std::vector<int> pixelOwnership(SCR_WIDTH*SCR_HEIGHT, -1);

std::chrono::steady_clock::time_point startTime = std::chrono::high_resolution_clock::now();

std::unordered_map<int, int> buildPixelCount(const std::vector<int>& pixelOwnership) 
{
    std::unordered_map<int, int> pixelCount;

    for (int i = 0; i < pixelOwnership.size(); i++) 
        if (pixelOwnership[i] != -1)
            pixelCount[pixelOwnership[i]]++;

    return pixelCount;
}

std::unordered_map<int, int> ownedPixelsMap = buildPixelCount(pixelOwnership);

void loadConfiguration() {
    SceneSettings settings = SceneConfigLoader::loadDefault();
    SceneConfigLoader::applyToGlobals(settings);
    
    SCR_WIDTH = settings.screenWidth;
    SCR_HEIGHT = settings.screenHeight;
    sphere_count = settings.sphereCount;
    cylinder_count = settings.cylinderCount;
    withOcclusionCulling = settings.withOcclusionCulling;
    topLevel = settings.downsampleLevel;
}

void renderFrameWithOcclusionCulling(std::vector<uint32_t>& framebuffer, 
    std::vector<float>& depthBuffer, 
    std::vector<CylinderIndex>& cylinders,
    std::vector<Sphere>& spheres,
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

    
    int frustumCylindersCount = 0;
    int visibleCylindersCount = 0;
    for(int i = 0; i < cylinders.size(); i++)
    {
        glm::vec3 pa(spheres[cylinders[i].sphereIndexA]);
        glm::vec3 pb(spheres[cylinders[i].sphereIndexB]);
        float cylRadius = cylinders[i].radius;
        if (frustum.isCylinderInside(pa, pb, cylRadius))
        {
            bool wasDrawn = drawCylinderWithOcclusionCulling(proj, view, up, front, right, camPos, SCR_WIDTH, SCR_HEIGHT, 
                pa, pb, cylRadius, 
                fov, framebuffer, depthBuffer, hizPyramid, cylinderVisibilityFrameCache[i], pixelOwnership, i + static_cast<int>(spheres.size()), ownedPixelsMap[i + static_cast<int>(spheres.size())]);
            frustumCylindersCount++;
            if (wasDrawn)
            {
                visibleCylindersCount++;
                cylinderVisibilityFrameCache[i] = cylinderVisibilityFrameCache[i] | 0b10000000;
            } 
        } else cylinderVisibilityFrameCache[i] = 0;
    }
    frustumCylinders = frustumCylindersCount;
    visibleCylinders = visibleCylindersCount;

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
    frustumSpheres = frustumSpheresCount;
    visibleSpheres = visibleSpheresCount;

    ownedPixelsMap = buildPixelCount(pixelOwnership);
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
    for(int i = 0; i < cylinders.size(); i++)
    {
        if (cylinderVisibilityFrameCache[i] & 0b10000000)
        {
            if (cylinderVisibilityLastFrameCache[i] <= (cylinderVisibilityFrameCache[i] & 0b01111111) &&
                ownedPixelsMap[i + static_cast<int>(spheres.size())] > 0) 
                cylinderVisibilityFrameCache[i] = 10;
        }
        cylinderVisibilityLastFrameCache[i] = cylinderVisibilityFrameCache[i] & 0b01111111;
    }
    std::fill(pixelOwnership.begin(), pixelOwnership.end(), -1);

    if (showHiz) drawMipmaps(hizPyramid, level, framebuffer, SCR_WIDTH, SCR_HEIGHT);
    
}

void renderFrameWithoutOcclusionCulling(std::vector<uint32_t>& framebuffer, 
    std::vector<float>& depthBuffer, 
    std::vector<CylinderIndex>& cylinders,
    std::vector<Sphere>& spheres,
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

    frustumCylinders = 0;
    for(int i = 0; i < cylinders.size(); i++)
    {
        glm::vec3 pa(spheres[cylinders[i].sphereIndexA]);
        glm::vec3 pb(spheres[cylinders[i].sphereIndexB]);
        float cylRadius = cylinders[i].radius;
        if (frustum.isCylinderInside(pa, pb, cylRadius))
        {
            frustumCylinders++;
            drawCylinder(proj, view, up, front, right, camPos, SCR_WIDTH, SCR_HEIGHT, 
                 pa, pb, cylRadius, 
                 fov, framebuffer, depthBuffer);
        } 
    }
    visibleCylinders = frustumCylinders;

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
    loadConfiguration();
    startTime = std::chrono::high_resolution_clock::now();
    std::unordered_map<std::string, int*> scene_data = {
        {"Screen width", &SCR_WIDTH},
        {"Screen height", &SCR_HEIGHT},
        {"Sphere count", &sphere_count},
        {"Spheres On Frustum", &frustumSpheres},
        {"Visible Spheres", &visibleSpheres},
        {"Cylinder count", &cylinder_count},
        {"Cylinders On Frustum", &frustumCylinders},
        {"Visible Cylinders", &visibleCylinders},
    };

    AppRenderer window { title + (withOcclusionCulling ? " - With Occlusion Culling" : " - Without Occlusion Culling"), SCR_WIDTH, SCR_HEIGHT, shown };
    Camera camera(SCR_WIDTH, SCR_HEIGHT);
    camera.SetPosition(.0f, .0f, .0f);
    CameraController camera_controller(window, camera);
    
    std::vector<uint32_t> framebuffer(SCR_WIDTH * SCR_HEIGHT);
    std::vector<float> depthBuffer(SCR_WIDTH * SCR_HEIGHT, FLT_MAX);
    
   
    std::vector<CylinderIndex> cylinders = getCylinderScene(cylinder_count);
    std::vector<Sphere> spheres = getScene(sphere_count);
    const int actualSphereCount = static_cast<int>(spheres.size());
    const int actualCylinderCount = static_cast<int>(cylinders.size());

    std::vector<std::pair<glm::vec3, glm::vec3>> chkPoints = getCheckpoints(sphere_count, spheres, cylinders);
    
    Benchmark benchmark(camera_controller, chkPoints);
    
    std::string base_path = withOcclusionCulling ? "media/csv/seq_w_cyl/occ_" : "media/csv/seq_w_cyl/"; 
    base_path += ((currentScene == SceneType::LOADED_SCENE) ? ("loaded_scene_" + scene_file) : "packed_scene");
    std::string frame_times_path = base_path + "_frame_times.csv";
    std::string process_times_path = base_path + "_process_times.csv";
    Profiler profiler(window, 
        frame_times_path, 
        process_times_path, 
        actualSphereCount, actualCylinderCount, (withOcclusionCulling ? topLevel : 0), 48.0f);
    
    window.setupSceneInfoGui("Scene Info", scene_data);
    window.setupCameraGui("Camera Info", &camera);
    window.setupInputInfoGui("General Input Info");
    window.setupBenchmarkInfoGui("Benchmark", &benchmark);

    std::chrono::steady_clock::time_point endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = endTime - startTime;
    std::ofstream logFile("C:\\Users\\Sebatian\\Desktop\\XLIM\\Memoria_per_frame\\log.txt", std::ios::out);
    logFile << title << std::endl;
    logFile << "Elapsed time: " << elapsed_seconds.count() << " seconds" << std::endl;
    logFile << "With occlusion culling: " << withOcclusionCulling << std::endl;
    logFile.close();

    std::fill(depthBuffer.begin(), depthBuffer.end(), FLT_MAX);
    hizPyramid = generateHiZPyramid(DepthBuffer{SCR_WIDTH, SCR_HEIGHT, camera.getFar(), depthBuffer}, topLevel);
    try
    {
        bool isRunning = true;
        while ( isRunning )
        {
            
            camera_controller.cameraUpdate();
            benchmark.update();

            profiler.updateProfiler(benchmark.getCheckpointID(), frustumSpheres, visibleSpheres, frustumCylinders, visibleCylinders);
            if (window.getInput().isKeyDown(Key::T)) profiler.startSavingNextFrames(benchmark.getCheckpointID());

            std::fill(framebuffer.begin(), framebuffer.end(), 0xFFFFFF00);
            
            if (withOcclusionCulling) 
            {
                renderFrameWithOcclusionCulling(framebuffer, depthBuffer, cylinders, spheres, camera);
                hizPyramid = generateHiZPyramid(DepthBuffer{SCR_WIDTH, SCR_HEIGHT, camera.getFar(), depthBuffer}, topLevel);
            } else 
                renderFrameWithoutOcclusionCulling(framebuffer, depthBuffer, cylinders, spheres, camera);

            std::fill(depthBuffer.begin(), depthBuffer.end(), FLT_MAX);

            window.updateTexture(framebuffer);

            isRunning = window.update();


            if (window.getInput().isKeyDown(Key::H)) 
            {
                std::cout << "Show hiz " << showHiz << std::endl;
                showHiz = !showHiz;
            }

            if (window.getInput().isKeyDown(Key::O)) 
            {
                withOcclusionCulling = !withOcclusionCulling;
                std::cout << "Occlusion Culling: " << (withOcclusionCulling ? "On" : "Off") << std::endl;
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




