#include <iostream>
#include <fstream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glad/glad.h>
#include <SDL3/SDL.h>
#include <vector>

#include "appSDLGL.h"

#include "geometry/cylinder/cylinder.h"

#include "utils/parallel/aux_functions.h"
#include "utils/benchmark_resources.h"

#include "ux/input.h"
#include "ux/camera_controller.h"
#include "ux/cinematic/benchmark.h"
#include "ux/profiler/profiler.h"

#include "vis/gl/frame_buffer.h"
#include "vis/gl/storage_buffer.h"
#include "vis/gl/texture.h"
#include "vis/canvas.h"
#include "vis/compute_shader_program.h"

using uint = unsigned int;

// Settings
int SCR_WIDTH = 1024;
int SCR_HEIGHT = 1024;
int sphere_count = 4000000;
int cylinder_count = 4000000;

std::string title = "First Parallel Version: Spheres and Cylinders - CPU Frustum Culling"; 

bool shown = true;
bool withOcclusionCulling = false;

GLuint workGroupSizeXPerPixel = 16;  // Deifining threads-per-group (X)
GLuint workGroupSizeYPerPixel = 16;  // Deifining threads-per-group (Y)

GLuint workGroupSizeXPerSphere = 256;  // Deifining threads-per-sphere (X)
GLuint workGroupSizeXPerCylinder = 256;  // Deifining threads-per-sphere (X)

int visibleSpheresCount = 0;
int notOccludedSpheresCount = 0;

int visibleCylindersCount = 0;
int notOccludedCylindersCount = 0;

// downsample settings
int downsampleLevel = 4;
int downsampleWorkGroupSizeX = 16/downsampleLevel;
int downsampleWorkGroupSizeY = 16/downsampleLevel;

std::chrono::steady_clock::time_point startTime = std::chrono::high_resolution_clock::now();

void mainWithoutOcclusionCulling(Camera& camera, AppOpenGL& window);

void mainWithOcclusionCulling(Camera& camera, AppOpenGL& window);


int main(int argc, char* argv[]) 
{
    startTime = std::chrono::high_resolution_clock::now();
    AppOpenGL window { title + (withOcclusionCulling ? " - With Occlusion Culling" : " - Without Occlusion Culling"), SCR_WIDTH, SCR_HEIGHT, shown };

    Camera camera(SCR_WIDTH, SCR_HEIGHT);
    camera.SetPosition(.0f, .0f, .0f);
   
    withOcclusionCulling ? 
        mainWithOcclusionCulling(camera, window) :
        mainWithoutOcclusionCulling(camera, window);

    return 0;
}

void mainWithOcclusionCulling(Camera& camera, AppOpenGL& window)
{
    std::unordered_map<std::string, int*> scene_data = {
        {"Screen width", &SCR_WIDTH},
        {"Screen height", &SCR_HEIGHT},
        {"Sphere count", &sphere_count},
        {"Spheres On Frustum", &visibleSpheresCount},
        {"Drawn spheres", &notOccludedSpheresCount},
        {"Cylinder count", &cylinder_count},
        {"Cylinders On Frustum", &visibleCylindersCount},
        {"Drawn cylinders", &notOccludedCylindersCount}
    };

    CameraController camera_controller(window, camera);

    ComputeShader spheresShader("assets/shaders/fst_parallel_attempt/cpu_cull/sphere.compute");
    ComputeShader cylinderShader("assets/shaders/fst_parallel_attempt/cpu_cull/cylinder.compute");

    ComputeShader cleaningComputeShader("assets/shaders/fst_parallel_attempt/cpu_cull/set_to_black.compute");
    ComputeShader hizPyramidComputeShader("assets/shaders/fst_parallel_attempt/cpu_cull/mipmap_gen.compute");
    ComputeShader pixelCountComputeShader("assets/shaders/fst_parallel_attempt/cpu_cull/pixel_count.compute");

    Canvas canvas(GL_TEXTURE_2D, GL_RGBA8, SCR_WIDTH, SCR_HEIGHT);
    canvas.setFBO(GL_COLOR_ATTACHMENT0);
    canvas.setupDepthData(SCR_WIDTH, SCR_HEIGHT);
    canvas.setupCleaningProgram(cleaningComputeShader);
    canvas.setupDepthDownsample(downsampleLevel, SCR_WIDTH, SCR_HEIGHT);
    canvas.setupDownsamplingProgram(hizPyramidComputeShader);

    StorageBuffer pixelCountFramesBuffer(GL_SHADER_STORAGE_BUFFER, SCR_WIDTH*SCR_HEIGHT * sizeof(GLuint), 4, 
        nullptr, GL_DYNAMIC_COPY);

    std::vector<Sphere> spheres = getScene(sphere_count);
    sphere_count = spheres.size(); 

    std::vector<Cylinder> cylinders = getCylinderScene(cylinder_count);

    cylinder_count = cylinders.size(); 

    std::vector<std::pair<glm::vec3, glm::vec3>> chkPoints = getCheckpoints(sphere_count, spheres, cylinders);
    
    std::vector<SphereContainer> visibleSpheres(sphere_count);    
    StorageBuffer sphereBuffer(GL_SHADER_STORAGE_BUFFER, sphere_count * sizeof(SphereContainer), 6, 
        visibleSpheres.data(), GL_STATIC_DRAW);

    std::vector<CylinderContainer> visibleCylinders(cylinder_count);    
    StorageBuffer cylinderBuffer(GL_SHADER_STORAGE_BUFFER, cylinder_count * sizeof(CylinderContainer), 7,
        visibleCylinders.data(), GL_STATIC_DRAW);
    
    // Framebuffer downsampleDepthFBO;
    // downsampleDepthFBO.attachTexture(GL_COLOR_ATTACHMENT0, downsampledDepthTexture);
    // all entities visibility info ssbo
    int totalEntities = sphere_count + cylinder_count;
    std::vector<GLuint> visibilityFrames(2 * totalEntities, 10);
    StorageBuffer visibilityFramesBuffer(GL_SHADER_STORAGE_BUFFER, 2 * totalEntities * sizeof(GLuint), 5, 
        visibilityFrames.data(), GL_DYNAMIC_COPY);
    
    GLuint zero = 0;
    GLuint resetValue = 0;
    StorageBuffer visibilityAtomicCounter(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint), 8, 
        &zero, GL_STATIC_DRAW);

    Benchmark benchmark(camera_controller, chkPoints);

    std::string base_path = "media/csv/fst_parallel_w_cyl/cpu_cull/occ_"; 
    base_path += ((currentScene == SceneType::LOADED_SCENE) ? ("loaded_scene_" + scene_file) : "packed_scene");
    std::string frame_times_path = base_path + "_frame_times.csv";
    std::string process_times_path = base_path + "_process_times.csv";
    Profiler profiler(window, 
        frame_times_path, 
        process_times_path, sphere_count, cylinder_count, downsampleLevel, 48.0f);


    window.setupSceneInfoGui("Scene Info", scene_data);
    window.setupCameraGui("Camera Info", &camera);
    window.setupInputInfoGui("General Input Info");
    window.setupBenchmarkInfoGui("Benchmark", &benchmark);
    
    // Calculating number of work groups (based on the number of threads and spheres)
    GLuint numGroupsXSpheres = (sphere_count + workGroupSizeXPerSphere - 1) / workGroupSizeXPerSphere;
    GLuint numGroupsXCylinders = (cylinder_count + workGroupSizeXPerCylinder - 1) / workGroupSizeXPerCylinder;
    GLuint numGroupsY = 1;

    canvas.bindTexture();
    canvas.bindFBO();

    glm::ivec2 screenResolution(SCR_WIDTH, SCR_HEIGHT);
    std::chrono::steady_clock::time_point endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = endTime - startTime;
    std::ofstream logFile("C:\\Users\\Sebatian\\Desktop\\XLIM\\Memoria_per_frame\\log.txt", std::ios::out);
    logFile << title << std::endl;
    logFile << "Elapsed time: " << elapsed_seconds.count() << " seconds" << std::endl;
    logFile << "With occlusion culling: " << withOcclusionCulling << std::endl;
    logFile.close();
    
    try
    {
        bool fbo1o2 = false;
        bool isRunning = true;
        while ( isRunning )
        {
            camera_controller.cameraUpdate();
            benchmark.update();
            profiler.updateProfiler(benchmark.getCheckpointID(), visibleSpheresCount, notOccludedSpheresCount, visibleCylindersCount, notOccludedCylindersCount);

            if (window.getInput().isKeyDown(Key::T)) profiler.startSavingNextFrames(benchmark.getCheckpointID());

            canvas.downsampleCanvasDepth(downsampleWorkGroupSizeX, downsampleWorkGroupSizeY);
            
            canvas.cleanCanvasBuffers(workGroupSizeXPerPixel, workGroupSizeYPerPixel, camera.getFar());

            Frustum frustum(camera);

            
            spheresShader.use();
            #if BENCHMARKING 
                visibilityAtomicCounter.bind();
                glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &notOccludedCylindersCount);
                glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
                visibilityAtomicCounter.unbind();
                spheresShader.setInt("benchmark", 1); 
            #else
                spheresShader.setInt("benchmark", 0);
            #endif
            cullSpheres(spheres, visibleSpheres, frustum, visibleSpheresCount);
            sphereBuffer.bind();
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, visibleSpheresCount * sizeof(SphereContainer), visibleSpheres.data());
            sphereBuffer.unbind();
            numGroupsXSpheres = (visibleSpheresCount + workGroupSizeXPerSphere - 1) / workGroupSizeXPerSphere;
            spheresShader.setInt("sphereCount", visibleSpheresCount); // visible sphere count given
            spheresShader.setUint("visibilityFrameBufferIndexOffset", 0);
            spheresShader.setVec2I("screenResolution", screenResolution);
            setCameraUniforms(spheresShader, camera);
            glDispatchCompute(numGroupsXSpheres, numGroupsY, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
            
            cylinderShader.use();
            #if BENCHMARKING
                visibilityAtomicCounter.bind();
                glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &notOccludedSpheresCount);
                glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
                visibilityAtomicCounter.unbind();
                cylinderShader.setInt("benchmark", 1);
            #else
                cylinderShader.setInt("benchmark", 0);
            #endif
            cullCylinders(cylinders, visibleCylinders, frustum, visibleCylindersCount);
            cylinderBuffer.bind();
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, visibleCylindersCount * sizeof(CylinderContainer), visibleCylinders.data());
            cylinderBuffer.unbind();
            numGroupsXCylinders = (visibleCylindersCount + workGroupSizeXPerSphere - 1) / workGroupSizeXPerSphere;
            cylinderShader.setInt("cylinderCount", visibleCylindersCount); // visible cylinder count given
            cylinderShader.setUint("visibilityFrameBufferIndexOffset", sphere_count);
            cylinderShader.setVec2I("screenResolution", screenResolution);
            setCameraUniforms(cylinderShader, camera);
            glDispatchCompute(numGroupsXCylinders, numGroupsY, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
            
            pixelCountComputeShader.use();
            pixelCountComputeShader.setVec2I("screenResolution", screenResolution);
            glDispatchCompute((SCR_WIDTH + workGroupSizeXPerPixel - 1) / workGroupSizeXPerPixel, 
                (SCR_HEIGHT + workGroupSizeYPerPixel - 1) / workGroupSizeYPerPixel, 
                1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            
            glBindFramebuffer(GL_READ_FRAMEBUFFER, canvas.getFramebuffer().getId());
            isRunning = window.update();

            //// if want mipmap check 
            // Blit from framebuffer to default framebuffer (screen)
            // if (fbo1o2)
            // {
            //     glBindFramebuffer(GL_READ_FRAMEBUFFER, canvas.getFramebuffer().getId());
            //     isRunning = window.update();
            // } 
            // else
            // {
            //     glBindFramebuffer(GL_READ_FRAMEBUFFER, downsampleDepthFBO.getId());
            //     isRunning = window.update(SCR_WIDTH/(1 << downsampleLevel), SCR_HEIGHT/(1 << downsampleLevel));
            // } 
            //// END:: if want mipmap check 
            
            if (window.getInput().isKeyDown(Key::F10)) 
            {
                std::string sshot_name  = "media/img/fst_parallel/frame_" + std::to_string(benchmark.getCheckpointID()) + ".bmp";
                canvas.takeScreenshot(sshot_name);
            }

            if (window.getInput().isKeyDown(Key::F)) fbo1o2 = !fbo1o2; 

        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return;
    }
}


void mainWithoutOcclusionCulling(Camera& camera, AppOpenGL& window)
{
    std::unordered_map<std::string, int*> scene_data = {
        {"Screen width", &SCR_WIDTH},
        {"Screen height", &SCR_HEIGHT},
        {"Sphere count", &sphere_count},
        {"Cylinder count", &cylinder_count},
        {"Spheres On Frustum", &visibleSpheresCount},
        {"Cylinders On Frustum", &visibleCylindersCount},
        {"Drawn spheres", &notOccludedSpheresCount},
        {"Drawn cylinders", &notOccludedCylindersCount}
    };

    CameraController camera_controller(window, camera);

    ComputeShader spheresShader("assets/shaders/fst_parallel_attempt/cpu_cull/sphere_no_occ.compute");
    ComputeShader cylinderShader("assets/shaders/fst_parallel_attempt/cpu_cull/cylinder_no_occ.compute");
    ComputeShader cleaningComputeShader("assets/shaders/fst_parallel_attempt/cpu_cull/set_to_black_no_occ.compute");

    Canvas canvas(GL_TEXTURE_2D, GL_RGBA8, SCR_WIDTH, SCR_HEIGHT);
    canvas.setFBO(GL_COLOR_ATTACHMENT0);
    canvas.setupDepthData(SCR_WIDTH, SCR_HEIGHT);
    canvas.setupCleaningProgram(cleaningComputeShader);


    std::vector<Sphere> spheres = getScene(sphere_count);
    sphere_count = spheres.size(); 

    std::vector<Cylinder> cylinders = getCylinderScene(cylinder_count);

    cylinder_count = cylinders.size(); 

    std::vector<std::pair<glm::vec3, glm::vec3>> chkPoints = getCheckpoints(sphere_count, spheres, cylinders);
    
    std::vector<SphereContainer> visibleSpheres(sphere_count);    
    StorageBuffer sphereBuffer(GL_SHADER_STORAGE_BUFFER, sphere_count * sizeof(SphereContainer), 6, 
        visibleSpheres.data(), GL_STATIC_DRAW);

    std::vector<CylinderContainer> visibleCylinders(cylinder_count);    
    StorageBuffer cylinderBuffer(GL_SHADER_STORAGE_BUFFER, cylinder_count * sizeof(CylinderContainer), 7,
        visibleCylinders.data(), GL_STATIC_DRAW);
    
    
    Benchmark benchmark(camera_controller, chkPoints);

    std::string base_path = "media/csv/fst_parallel_w_cyl/cpu_cull/"; 
    base_path += ((currentScene == SceneType::LOADED_SCENE) ? ("loaded_scene_" + scene_file) : "packed_scene");
    std::string frame_times_path = base_path + "_frame_times.csv";
    std::string process_times_path = base_path + "_process_times.csv";
    Profiler profiler(window, 
        frame_times_path, 
        process_times_path, sphere_count, cylinder_count, 0, 48.0f);


    window.setupSceneInfoGui("Scene Info", scene_data);
    window.setupCameraGui("Camera Info", &camera);
    window.setupInputInfoGui("General Input Info");
    window.setupBenchmarkInfoGui("Benchmark", &benchmark);
    
    // Calculating number of work groups (based on the number of threads and spheres)
    GLuint numGroupsXSpheres = (sphere_count + workGroupSizeXPerSphere - 1) / workGroupSizeXPerSphere;
    GLuint numGroupsXCylinders = (cylinder_count + workGroupSizeXPerCylinder - 1) / workGroupSizeXPerCylinder;
    GLuint numGroupsY = 1;

    canvas.bindTexture();
    canvas.bindFBO();

    glm::ivec2 screenResolution(SCR_WIDTH, SCR_HEIGHT);

    std::chrono::steady_clock::time_point endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = endTime - startTime;
    std::ofstream logFile("C:\\Users\\Sebatian\\Desktop\\XLIM\\Memoria_per_frame\\log.txt", std::ios::out);
    logFile << title << std::endl;
    logFile << "Elapsed time: " << elapsed_seconds.count() << " seconds" << std::endl;
    logFile << "With occlusion culling: " << withOcclusionCulling << std::endl;
    logFile.close();
    try
    {
        bool isRunning = true;
        while ( isRunning )
        {
            camera_controller.cameraUpdate();
            benchmark.update();
            profiler.updateProfiler(benchmark.getCheckpointID(), visibleSpheresCount, notOccludedSpheresCount, visibleCylindersCount, notOccludedCylindersCount);

            if (window.getInput().isKeyDown(Key::T)) profiler.startSavingNextFrames(benchmark.getCheckpointID());
            
            canvas.cleanCanvasBuffers(workGroupSizeXPerPixel, workGroupSizeYPerPixel, camera.getFar());

            Frustum frustum(camera);
            
            spheresShader.use();
            cullSpheres(spheres, visibleSpheres, frustum, visibleSpheresCount);
            notOccludedSpheresCount = visibleSpheresCount;
            sphereBuffer.bind();
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, visibleSpheresCount * sizeof(SphereContainer), visibleSpheres.data());
            sphereBuffer.unbind();
            numGroupsXSpheres = (visibleSpheresCount + workGroupSizeXPerSphere - 1) / workGroupSizeXPerSphere;
            spheresShader.setInt("sphereCount", visibleSpheresCount); // visible sphere count given
            spheresShader.setVec2I("screenResolution", screenResolution);
            setCameraUniforms(spheresShader, camera);
            glDispatchCompute(numGroupsXSpheres, numGroupsY, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
            
            cylinderShader.use();
            cullCylinders(cylinders, visibleCylinders, frustum, visibleCylindersCount);
            notOccludedCylindersCount = visibleCylindersCount;
            cylinderBuffer.bind();
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, visibleCylindersCount * sizeof(CylinderContainer), visibleCylinders.data());
            cylinderBuffer.unbind();
            numGroupsXCylinders = (visibleCylindersCount + workGroupSizeXPerSphere - 1) / workGroupSizeXPerSphere;
            cylinderShader.setInt("cylinderCount", visibleCylindersCount); // visible cylinder count given
            cylinderShader.setVec2I("screenResolution", screenResolution);
            setCameraUniforms(cylinderShader, camera);
            glDispatchCompute(numGroupsXCylinders, numGroupsY, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
            
            
            glBindFramebuffer(GL_READ_FRAMEBUFFER, canvas.getFramebuffer().getId());
            isRunning = window.update();
            
            if (window.getInput().isKeyDown(Key::F10)) 
            {
                std::string sshot_name  = "media/img/fst_parallel/frame_" + std::to_string(benchmark.getCheckpointID()) + ".bmp";
                canvas.takeScreenshot(sshot_name);
            }

        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return;
    }
}