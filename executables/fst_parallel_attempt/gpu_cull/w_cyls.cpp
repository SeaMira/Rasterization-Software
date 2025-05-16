#include <iostream>
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
int sphere_count = 100000000;
int cylinder_count = 0;

std::string title = "First Parallel Version GPU culling: Spheres and Cylinders"; 

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

void mainWithoutOcclusionCulling(Camera& camera, AppOpenGL& window);

void mainWithOcclusionCulling(Camera& camera, AppOpenGL& window);

int main(int argc, char* argv[]) 
{
    AppOpenGL window { title, SCR_WIDTH, SCR_HEIGHT, shown };

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
        {"Cylinder count", &cylinder_count},
        {"Spheres On Frustum", &visibleSpheresCount},
        {"Cylinders On Frustum", &visibleCylindersCount},
        {"Drawn spheres", &notOccludedSpheresCount}
    };

    CameraController camera_controller(window, camera);
    
    ComputeShader spheresShader("assets/shaders/fst_parallel_attempt/gpu_cull/sphere_culling.compute");
    ComputeShader cylinderShader("assets/shaders/fst_parallel_attempt/gpu_cull/cylinder_culling.compute");
    
    ComputeShader cleaningComputeShader("assets/shaders/fst_parallel_attempt/gpu_cull/set_to_black.compute");
    ComputeShader hizPyramidComputeShader("assets/shaders/fst_parallel_attempt/gpu_cull/mipmap_gen.compute");
    ComputeShader pixelCountComputeShader("assets/shaders/fst_parallel_attempt/gpu_cull/pixel_count.compute");

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
    std::vector<CylinderContainer> visibleCylinders(cylinder_count);  

    visibleSpheresCount = sphere_count;
    // fillSpheresData(spheres, visibleSpheres);
    
    visibleCylindersCount = cylinder_count;
    // fillCylindersData(cylinders, visibleCylinders, sphere_count);
    
    StorageBuffer sphereBuffer(GL_SHADER_STORAGE_BUFFER, sphere_count * sizeof(Sphere), 6, 
        spheres.data(), GL_STATIC_DRAW);
    
    StorageBuffer cylinderBuffer(GL_SHADER_STORAGE_BUFFER, cylinder_count * sizeof(Cylinder), 7, 
        cylinders.data(), GL_STATIC_DRAW);
    
    // Framebuffer downsampleDepthFBO;
    // downsampleDepthFBO.attachTexture(GL_COLOR_ATTACHMENT0, downsampledDepthTexture);
    // all entities visibility info ssbo
    int totalEntities = sphere_count + cylinder_count;
    std::vector<GLuint> visibilityFrames(2 * totalEntities, 10);
    StorageBuffer visibilityFramesBuffer(GL_SHADER_STORAGE_BUFFER, 2 * totalEntities * sizeof(GLuint), 5, 
        visibilityFrames.data(), GL_DYNAMIC_COPY);
    
    GLuint zero = 0;
    GLuint resetValue = 0;
    StorageBuffer frustumAtomicCounter(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint), 8, 
        &zero, GL_STATIC_DRAW);
    StorageBuffer occlusionAtomicCounter(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint), 9, 
        &zero, GL_STATIC_DRAW);
    
    Benchmark benchmark(camera_controller, chkPoints);

    std::string base_path = "media/csv/fst_parallel_w_cyl/occ_"; 
    base_path += ((currentScene == SceneType::LOADED_SCENE) ? ("loaded_scene_" + scene_file) : "packed_scene");
    std::string frame_times_path = base_path + "_gpu_frame_times.csv";
    std::string process_times_path = base_path + "_gpu_frame_times.csv";
    Profiler profiler(window, 
        frame_times_path, 
        process_times_path, sphere_count, cylinder_count, downsampleLevel);

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
                frustumAtomicCounter.bind();
                glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &visibleCylindersCount);
                glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
                frustumAtomicCounter.unbind();

                occlusionAtomicCounter.bind();
                glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &notOccludedCylindersCount);
                glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
                occlusionAtomicCounter.unbind();

                spheresShader.setInt("benchmark", 1); 
            #else
                spheresShader.setInt("benchmark", 0);
            #endif
            spheresShader.setInt("sphereCount", sphere_count);
            spheresShader.setUint("visibilityFrameBufferIndexOffset", 0);
            spheresShader.setVec2I("screenResolution", screenResolution);
            setFrustumUniforms(spheresShader, frustum);
            setCameraUniforms(spheresShader, camera);
            glDispatchCompute(numGroupsXSpheres, numGroupsY, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
            
            cylinderShader.use();
            #if BENCHMARKING 
                frustumAtomicCounter.bind();
                glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &visibleSpheresCount);
                glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
                frustumAtomicCounter.unbind();

                occlusionAtomicCounter.bind();
                glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &notOccludedSpheresCount);
                glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
                occlusionAtomicCounter.unbind();
                
                cylinderShader.setInt("benchmark", 1); 
            #else
                cylinderShader.setInt("benchmark", 0);
            #endif
            cylinderShader.setInt("cylinderCount", cylinder_count);
            cylinderShader.setUint("visibilityFrameBufferIndexOffset", sphere_count);
            cylinderShader.setVec2I("screenResolution", screenResolution);
            setFrustumUniforms(cylinderShader, frustum);
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
        return ;
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
        {"Drawn spheres", &notOccludedSpheresCount}
    };

    CameraController camera_controller(window, camera);
    
    ComputeShader spheresShader("assets/shaders/fst_parallel_attempt/gpu_cull/sphere_culling_no_occ.compute");
    ComputeShader cylinderShader("assets/shaders/fst_parallel_attempt/gpu_cull/cylinder_culling_no_occ.compute");
    ComputeShader cleaningComputeShader("assets/shaders/fst_parallel_attempt/gpu_cull/set_to_black_no_occ.compute");
    
    Canvas canvas(GL_TEXTURE_2D, GL_RGBA8, SCR_WIDTH, SCR_HEIGHT);
    canvas.setFBO(GL_COLOR_ATTACHMENT0);
    canvas.setupDepthData(SCR_WIDTH, SCR_HEIGHT);
    canvas.setupCleaningProgram(cleaningComputeShader);
    
    std::vector<Sphere> spheres = getScene(sphere_count);
    sphere_count = spheres.size(); 

    std::vector<Cylinder> cylinders = getCylinderScene(cylinder_count);
    cylinder_count = cylinders.size(); 

    std::vector<std::pair<glm::vec3, glm::vec3>> chkPoints = getCheckpoints(sphere_count, spheres, cylinders);
    
    visibleSpheresCount = sphere_count;
    
    visibleCylindersCount = cylinder_count;
    
    StorageBuffer sphereBuffer(GL_SHADER_STORAGE_BUFFER, sphere_count * sizeof(Sphere), 6, 
        spheres.data(), GL_STATIC_DRAW);
    
    StorageBuffer cylinderBuffer(GL_SHADER_STORAGE_BUFFER, cylinder_count * sizeof(Cylinder), 7, 
        cylinders.data(), GL_STATIC_DRAW);
    

    GLuint zero = 0;
    GLuint resetValue = 0;
    StorageBuffer frustumAtomicCounter(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint), 8, 
        &zero, GL_STATIC_DRAW);

    Benchmark benchmark(camera_controller, chkPoints);

    std::string base_path = "media/csv/fst_parallel_w_cyl/"; 
    base_path += ((currentScene == SceneType::LOADED_SCENE) ? ("loaded_scene_" + scene_file) : "packed_scene");
    std::string frame_times_path = base_path + "_gpu_frame_times.csv";
    std::string process_times_path = base_path + "_gpu_frame_times.csv";
    Profiler profiler(window, 
        frame_times_path, 
        process_times_path, sphere_count, cylinder_count, 0);

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
            #if BENCHMARKING 
                frustumAtomicCounter.bind();
                glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &notOccludedCylindersCount);
                glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
                frustumAtomicCounter.unbind();
                spheresShader.setInt("benchmark", 1); 
            #else
                spheresShader.setInt("benchmark", 0);
            #endif
            spheresShader.setInt("sphereCount", sphere_count);
            spheresShader.setVec2I("screenResolution", screenResolution);
            setFrustumUniforms(spheresShader, frustum);
            setCameraUniforms(spheresShader, camera);
            glDispatchCompute(numGroupsXSpheres, numGroupsY, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
            
            cylinderShader.use();
            #if BENCHMARKING 
                frustumAtomicCounter.bind();
                glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &notOccludedCylindersCount);
                glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
                frustumAtomicCounter.unbind();
                cylinderShader.setInt("benchmark", 1); 
            #else
                cylinderShader.setInt("benchmark", 0);
            #endif
            cylinderShader.setInt("cylinderCount", cylinder_count);
            cylinderShader.setVec2I("screenResolution", screenResolution);
            setFrustumUniforms(cylinderShader, frustum);
            setCameraUniforms(cylinderShader, camera);
            glDispatchCompute(numGroupsXCylinders, numGroupsY, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
            
            glBindFramebuffer(GL_READ_FRAMEBUFFER, canvas.getFramebuffer().getId());
            isRunning = window.update();

            ////// checking drawn spheres ///////
            
            // sphereBuffer.bind();  // Primero aseguramos que el buffer está vinculado

            // // mapping buffer on reading mode
            // void* mappedData = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 
            //                                     0, 
            //                                     spheres.size() * sizeof(SphereContainer), 
            //                                     GL_MAP_READ_BIT);

            // // verifying correct mapping
            // if (mappedData != nullptr) {
            //     // Accessing mapped buffer
            //     SphereContainer* spheresData = reinterpret_cast<SphereContainer*>(mappedData);
            //     notOccludedSpheresCount = 0;
            //     // looping on elements checking if drawn or not
            //     for (size_t i = 0; i < visibleSpheresCount; ++i)
            //         if (spheresData[i].wasDrawn[0] == 1) 
            //             notOccludedSpheresCount++;
                
            //     // unmapping buffer once finished
            //     glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
            // } else {
            //     std::cerr << "Failed to map the buffer!" << std::endl;
            // }
            
            ////// END:: checking drawn spheres ///////
            
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
        return ;
    }
}