#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glad/glad.h>
#include <SDL3/SDL.h>
#include <vector>

#include "appSDLGL.h"

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
int sphere_count = 0;
int cylinder_count = 150;

std::string title = "Second Parallel Version"; 

bool shown = true;
bool withOcclusionCulling = true;

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

struct SphereBillboard 
{
    glm::vec4 sphPosR;   // 16b - camera space position of the sphere and radius
    glm::vec2 minCorner;     // 16b - up right corner of the billboard in camera space and the max X coordinate in screen space 
    glm::vec2 maxCorner;      // 16b - up left corner of the billboard in camera space and the max Y coordinate in screen space
    uint index;
    int padding1, padding2, padding3;
    // 96b
}; 

struct CylinderBillboard 
{
    glm::vec4 pa_r;   
    glm::vec4 pb_r;   
    glm::vec2 corner0;     
    glm::vec2 corner1;     
    glm::vec2 corner2;     
    glm::vec2 corner3;     
    uint index;
    int padding1, padding2, padding3;
}; 

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
        {"Spheres On Frustum", &visibleSpheresCount},
        {"Drawn spheres", &notOccludedSpheresCount},
        {"Cylinder count", &cylinder_count},
        {"Cylinders On Frustum", &visibleCylindersCount},
        {"Drawn cylinders", &notOccludedCylindersCount}
    };

    CameraController camera_controller(window, camera);
    
    ComputeShader cleaningComputeShader("assets/shaders/snd_parallel_attempt/cpu_cull/set_to_black.compute");
    
    ComputeShader sphBboxExtractionShader("assets/shaders/snd_parallel_attempt/cpu_cull/sph_bbox_ext.compute");
    ComputeShader sphBboxIntersectionShader("assets/shaders/snd_parallel_attempt/cpu_cull/sph_bbox_int.compute");
    
    ComputeShader cylBboxExtractionShader("assets/shaders/snd_parallel_attempt/cpu_cull/cyl_bbox_ext.compute");
    ComputeShader cylBboxIntersectionShader("assets/shaders/snd_parallel_attempt/cpu_cull/cyl_bbox_int.compute");

    ComputeShader hizPyramidComputeShader("assets/shaders/snd_parallel_attempt/cpu_cull/mipmap_gen.compute");
    ComputeShader pixelCountComputeShader("assets/shaders/snd_parallel_attempt/cpu_cull/pixel_count.compute");

    Canvas canvas(GL_TEXTURE_2D, GL_RGBA8, SCR_WIDTH, SCR_HEIGHT);
    canvas.setFBO(GL_COLOR_ATTACHMENT0);
    canvas.setupDepthData(SCR_WIDTH, SCR_HEIGHT);
    canvas.setupCleaningProgram(cleaningComputeShader);
    canvas.setupDepthDownsample(downsampleLevel, SCR_WIDTH, SCR_HEIGHT);
    canvas.setupDownsamplingProgram(hizPyramidComputeShader);
        
    StorageBuffer pixelCountFramesBuffer(GL_SHADER_STORAGE_BUFFER, SCR_WIDTH*SCR_HEIGHT * sizeof(GLuint), 4, 
        nullptr, GL_DYNAMIC_COPY);

    // SPHERES
    std::vector<glm::vec4> spheres = getScene(sphere_count);
    sphere_count = spheres.size(); 
    std::vector<Cylinder> cylinders = getCylinderScene(cylinder_count);
    cylinder_count = cylinders.size();
    std::vector<std::pair<glm::vec3, glm::vec3>> chkPoints = getCheckpoints(sphere_count, spheres, cylinders);

    std::vector<SphereContainer> visibleSpheres(sphere_count);    
    StorageBuffer sphereBuffer(GL_SHADER_STORAGE_BUFFER, sphere_count * sizeof(SphereContainer), 6, 
    visibleSpheres.data(), GL_STATIC_DRAW);
    
    StorageBuffer sphereBillboardBuffer(GL_SHADER_STORAGE_BUFFER, sphere_count * sizeof(SphereBillboard), 7,
        nullptr, GL_STATIC_DRAW);
    //////////

    // CYLINDERS

    std::vector<CylinderContainer> visibleCylinders(cylinder_count);    
    StorageBuffer cylinderBuffer(GL_SHADER_STORAGE_BUFFER, cylinder_count * sizeof(CylinderContainer), 9,
        visibleCylinders.data(), GL_STATIC_DRAW);

    StorageBuffer cylinderBillboardBuffer(GL_SHADER_STORAGE_BUFFER, cylinder_count * sizeof(CylinderBillboard), 10,
        nullptr, GL_STATIC_DRAW);
    //////////

    GLuint zero = 0;
    GLuint resetValue = 0;
    StorageBuffer visibilityAtomicCounter(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint), 8, 
        &zero, GL_STATIC_DRAW);

    int totalEntities = sphere_count + cylinder_count;
    std::vector<GLuint> visibilityFrames(2 * totalEntities, 10);
    StorageBuffer visibilityFramesBuffer(GL_SHADER_STORAGE_BUFFER, 2 * totalEntities * sizeof(GLuint), 5, 
    visibilityFrames.data(), GL_DYNAMIC_COPY);
    

    Benchmark benchmark(camera_controller, chkPoints);

    std::string base_path = "media/csv/scnd_parallel_w_cyl/occ_"; 
    base_path += ((currentScene == SceneType::LOADED_SCENE) ? ("loaded_scene_" + scene_file) : "packed_scene");
    std::string frame_times_path = base_path + "_cpu_frame_times.csv";
    std::string process_times_path = base_path + "_cpu_frame_times.csv";
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
        bool isRunning = true;
        while ( isRunning )
        {
            camera_controller.cameraUpdate();
            benchmark.update();
            profiler.updateProfiler(benchmark.getCheckpointID(), visibleSpheresCount, notOccludedSpheresCount, visibleCylindersCount, notOccludedCylindersCount);
            
            if (window.getInput().isKeyDown(Key::T)) profiler.startSavingNextFrames(benchmark.getCheckpointID());

            canvas.downsampleCanvasDepth(downsampleWorkGroupSizeX, downsampleWorkGroupSizeY);
            
            canvas.cleanCanvasBuffers(workGroupSizeXPerPixel, workGroupSizeYPerPixel, camera.getFar());

            visibilityAtomicCounter.bind();
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &notOccludedCylindersCount);
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
            visibilityAtomicCounter.unbind();

            Frustum frustum(camera);

            ///// SPHERES //////////
            // bbox extraction shader
            sphBboxExtractionShader.use();
            cullSpheres(spheres, visibleSpheres, frustum, visibleSpheresCount);
            sphereBuffer.bind();
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, visibleSpheresCount * sizeof(SphereContainer), visibleSpheres.data());
            sphereBuffer.unbind();
            numGroupsXSpheres = (visibleSpheresCount + workGroupSizeXPerSphere - 1) / workGroupSizeXPerSphere;
            sphBboxExtractionShader.setInt("sphereCount", visibleSpheresCount); // visible sphere count given
            sphBboxExtractionShader.setUint("visibilityFrameBufferIndexOffset", 0);
            sphBboxExtractionShader.setVec2I("screenResolution", screenResolution);
            setCameraUniforms(sphBboxExtractionShader, camera);
            glDispatchCompute(numGroupsXSpheres, numGroupsY, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            
            // bbox intersection shader
            sphBboxIntersectionShader.use();
            sphBboxIntersectionShader.setFloat("far", camera.getFar());
            sphBboxIntersectionShader.setVec2I("screenResolution", screenResolution);
            setCameraUniforms(sphBboxIntersectionShader, camera);
            glDispatchCompute((SCR_WIDTH + workGroupSizeXPerPixel - 1) / workGroupSizeXPerPixel, 
                (SCR_HEIGHT + workGroupSizeYPerPixel - 1) / workGroupSizeYPerPixel, 
                1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

            ///// SPHERES //////////

            visibilityAtomicCounter.bind();
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &notOccludedSpheresCount);
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
            visibilityAtomicCounter.unbind();

            ///// CYLINDERS //////////
            cylBboxExtractionShader.use();
            cullCylinders(cylinders, visibleCylinders, frustum, visibleCylindersCount);
            cylinderBuffer.bind();
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, visibleCylindersCount * sizeof(CylinderContainer), visibleCylinders.data());
            cylinderBuffer.unbind();
            numGroupsXCylinders = (visibleCylindersCount + workGroupSizeXPerCylinder - 1) / workGroupSizeXPerCylinder;
            cylBboxExtractionShader.setInt("cylinderCount", visibleCylindersCount); // visible sphere count given
            cylBboxExtractionShader.setUint("visibilityFrameBufferIndexOffset", sphere_count);
            cylBboxExtractionShader.setVec2I("screenResolution", screenResolution);
            setCameraUniforms(cylBboxExtractionShader, camera);
            glDispatchCompute(numGroupsXCylinders, numGroupsY, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            
            // bbox intersection shader
            cylBboxIntersectionShader.use();
            cylBboxIntersectionShader.setFloat("far", camera.getFar());
            cylBboxIntersectionShader.setVec2I("screenResolution", screenResolution);
            setCameraUniforms(cylBboxIntersectionShader, camera);
            glDispatchCompute((SCR_WIDTH + workGroupSizeXPerPixel - 1) / workGroupSizeXPerPixel, 
                (SCR_HEIGHT + workGroupSizeYPerPixel - 1) / workGroupSizeYPerPixel, 
                1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
            ///// CYLINDERS //////////

            pixelCountComputeShader.use();
            pixelCountComputeShader.setVec2I("screenResolution", screenResolution);
            glDispatchCompute((SCR_WIDTH + workGroupSizeYPerPixel - 1) / workGroupSizeYPerPixel, 
                (SCR_HEIGHT + workGroupSizeYPerPixel - 1) / workGroupSizeYPerPixel, 
                1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            
            glBindFramebuffer(GL_READ_FRAMEBUFFER, canvas.getFramebuffer().getId());
            isRunning = window.update();

            if (window.getInput().isKeyDown(Key::F10)) 
            {
                std::string sshot_name  = "media/img/scnd_parallel/frame_" + std::to_string(benchmark.getCheckpointID()) + ".bmp";
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


void mainWithoutOcclusionCulling(Camera& camera, AppOpenGL& window)
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
    
    ComputeShader cleaningComputeShader("assets/shaders/snd_parallel_attempt/cpu_cull/set_to_black_no_occ.compute");
    
    ComputeShader sphBboxExtractionShader("assets/shaders/snd_parallel_attempt/cpu_cull/sph_bbox_ext_no_occ.compute");
    ComputeShader sphBboxIntersectionShader("assets/shaders/snd_parallel_attempt/cpu_cull/sph_bbox_int_no_occ.compute");
    
    ComputeShader cylBboxExtractionShader("assets/shaders/snd_parallel_attempt/cpu_cull/cyl_bbox_ext_no_occ.compute");
    ComputeShader cylBboxIntersectionShader("assets/shaders/snd_parallel_attempt/cpu_cull/cyl_bbox_int_no_occ.compute");

    Canvas canvas(GL_TEXTURE_2D, GL_RGBA8, SCR_WIDTH, SCR_HEIGHT);
    canvas.setFBO(GL_COLOR_ATTACHMENT0);
    canvas.setupDepthData(SCR_WIDTH, SCR_HEIGHT);
    canvas.setupCleaningProgram(cleaningComputeShader);

    // SPHERES
    std::vector<glm::vec4> spheres = getScene(sphere_count);
    sphere_count = spheres.size(); 
    std::vector<Cylinder> cylinders = getCylinderScene(cylinder_count);
    cylinder_count = cylinders.size();
    std::vector<std::pair<glm::vec3, glm::vec3>> chkPoints = getCheckpoints(sphere_count, spheres, cylinders);

    std::vector<SphereContainer> visibleSpheres(sphere_count);    
    StorageBuffer sphereBuffer(GL_SHADER_STORAGE_BUFFER, sphere_count * sizeof(SphereContainer), 6, 
    visibleSpheres.data(), GL_STATIC_DRAW);
    
    StorageBuffer sphereBillboardBuffer(GL_SHADER_STORAGE_BUFFER, sphere_count * sizeof(SphereBillboard), 7,
        nullptr, GL_STATIC_DRAW);
    //////////

    // CYLINDERS

    std::vector<CylinderContainer> visibleCylinders(cylinder_count);    
    StorageBuffer cylinderBuffer(GL_SHADER_STORAGE_BUFFER, cylinder_count * sizeof(CylinderContainer), 9,
        visibleCylinders.data(), GL_STATIC_DRAW);

    StorageBuffer cylinderBillboardBuffer(GL_SHADER_STORAGE_BUFFER, cylinder_count * sizeof(CylinderBillboard), 10,
        nullptr, GL_STATIC_DRAW);
    //////////
    
    Benchmark benchmark(camera_controller, chkPoints);

    std::string base_path = "media/csv/scnd_parallel_w_cyl/"; 
    base_path += ((currentScene == SceneType::LOADED_SCENE) ? ("loaded_scene_" + scene_file) : "packed_scene");
    std::string frame_times_path = base_path + "_cpu_frame_times.csv";
    std::string process_times_path = base_path + "_cpu_frame_times.csv";
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
        bool isRunning = true;
        while ( isRunning )
        {
            camera_controller.cameraUpdate();
            benchmark.update();
            profiler.updateProfiler(benchmark.getCheckpointID(), visibleSpheresCount, notOccludedSpheresCount, visibleCylindersCount, notOccludedCylindersCount);
            
            if (window.getInput().isKeyDown(Key::T)) profiler.startSavingNextFrames(benchmark.getCheckpointID());
            
            canvas.cleanCanvasBuffers(workGroupSizeXPerPixel, workGroupSizeYPerPixel, camera.getFar());

            Frustum frustum(camera);

            ///// SPHERES //////////
            // bbox extraction shader
            sphBboxExtractionShader.use();
            cullSpheres(spheres, visibleSpheres, frustum, visibleSpheresCount);
            notOccludedSpheresCount = visibleSpheresCount;
            sphereBuffer.bind();
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, visibleSpheresCount * sizeof(SphereContainer), visibleSpheres.data());
            sphereBuffer.unbind();
            numGroupsXSpheres = (visibleSpheresCount + workGroupSizeXPerSphere - 1) / workGroupSizeXPerSphere;
            sphBboxExtractionShader.setInt("sphereCount", visibleSpheresCount); // visible sphere count given
            sphBboxExtractionShader.setVec2I("screenResolution", screenResolution);
            setCameraUniforms(sphBboxExtractionShader, camera);
            glDispatchCompute(numGroupsXSpheres, numGroupsY, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            
            // bbox intersection shader
            sphBboxIntersectionShader.use();
            sphBboxIntersectionShader.setFloat("far", camera.getFar());
            sphBboxIntersectionShader.setInt("sphereCount", visibleSpheresCount);
            sphBboxIntersectionShader.setVec2I("screenResolution", screenResolution);
            setCameraUniforms(sphBboxIntersectionShader, camera);
            glDispatchCompute((SCR_WIDTH + workGroupSizeXPerPixel - 1) / workGroupSizeXPerPixel, 
                (SCR_HEIGHT + workGroupSizeYPerPixel - 1) / workGroupSizeYPerPixel, 
                1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

            ///// SPHERES //////////

            ///// CYLINDERS //////////
            cylBboxExtractionShader.use();
            cullCylinders(cylinders, visibleCylinders, frustum, visibleCylindersCount);
            notOccludedCylindersCount = visibleCylindersCount;
            cylinderBuffer.bind();
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, visibleCylindersCount * sizeof(CylinderContainer), visibleCylinders.data());
            cylinderBuffer.unbind();
            numGroupsXCylinders = (visibleCylindersCount + workGroupSizeXPerCylinder - 1) / workGroupSizeXPerCylinder;
            cylBboxExtractionShader.setInt("cylinderCount", visibleCylindersCount); // visible sphere count given
            cylBboxExtractionShader.setVec2I("screenResolution", screenResolution);
            setCameraUniforms(cylBboxExtractionShader, camera);
            glDispatchCompute(numGroupsXCylinders, numGroupsY, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            
            // bbox intersection shader
            cylBboxIntersectionShader.use();
            cylBboxIntersectionShader.setFloat("far", camera.getFar());
            cylBboxIntersectionShader.setInt("cylinderCount", visibleCylindersCount);
            cylBboxIntersectionShader.setVec2I("screenResolution", screenResolution);
            setCameraUniforms(cylBboxIntersectionShader, camera);
            glDispatchCompute((SCR_WIDTH + workGroupSizeXPerPixel - 1) / workGroupSizeXPerPixel, 
                (SCR_HEIGHT + workGroupSizeYPerPixel - 1) / workGroupSizeYPerPixel, 
                1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
            ///// CYLINDERS //////////
            
            glBindFramebuffer(GL_READ_FRAMEBUFFER, canvas.getFramebuffer().getId());
            isRunning = window.update();

            if (window.getInput().isKeyDown(Key::F10)) 
            {
                std::string sshot_name  = "media/img/scnd_parallel/frame_" + std::to_string(benchmark.getCheckpointID()) + ".bmp";
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