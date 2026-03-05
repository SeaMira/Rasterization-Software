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
#include "utils/scene_config_loader.h"

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
int sphere_count = 10000;

std::string title = "Second Parallel Version"; 

bool shown = true;

GLuint workGroupSizeXPerPixel = 16;  // Deifining threads-per-group (X)
GLuint workGroupSizeYPerPixel = 16;  // Deifining threads-per-group (Y)

GLuint workGroupSizeXPerSphere = 256;  // Deifining threads-per-sphere (X)

int visibleSpheresCount = 0;
int notOccludedSpheresCount = 0;

// downsample settings
int downsampleLevel = 4;
int downsampleWorkGroupSizeX = 16/downsampleLevel;
int downsampleWorkGroupSizeY = 16/downsampleLevel;

double timerDuration = 48.0;

void loadConfiguration()
{
    SceneSettings settings = SceneConfigLoader::loadDefault();
    SceneConfigLoader::applyToGlobals(settings);
    
    SCR_WIDTH = settings.screenWidth;
    SCR_HEIGHT = settings.screenHeight;
    sphere_count = settings.sphereCount;
    downsampleLevel = settings.downsampleLevel;
    workGroupSizeXPerPixel = settings.workGroupSizePerPixelX;
    workGroupSizeYPerPixel = settings.workGroupSizePerPixelY;
    workGroupSizeXPerSphere = settings.workGroupSizePerSphere;
    timerDuration = settings.timerDuration;
    
    downsampleWorkGroupSizeX = 16/downsampleLevel;
    downsampleWorkGroupSizeY = 16/downsampleLevel;
}

struct SphereBillboard 
{
    glm::vec4 sphPosR;   // 16b - camera space position of the sphere and radius
    glm::vec2 minCorner;     // 16b - up right corner of the billboard in camera space and the max X coordinate in screen space 
    glm::vec2 maxCorner;      // 16b - up left corner of the billboard in camera space and the max Y coordinate in screen space
    uint index;
    int padding1, padding2, padding3;
    // 96b
}; 

int main(int argc, char* argv[]) 
{
    loadConfiguration();
    
    std::unordered_map<std::string, int*> scene_data = {
        {"Screen width", &SCR_WIDTH},
        {"Screen height", &SCR_HEIGHT},
        {"Sphere count", &sphere_count},
        {"Spheres On Frustum", &visibleSpheresCount},
        {"Drawn spheres", &notOccludedSpheresCount}
    };

    AppOpenGL window { title, SCR_WIDTH, SCR_HEIGHT, shown };
    Camera camera(SCR_WIDTH, SCR_HEIGHT);
    camera.SetPosition(.0f, .0f, .0f);
    CameraController camera_controller(window, camera);
    
    ComputeShader cleaningComputeShader("assets/shaders/snd_parallel_attempt/cpu_cull/set_to_black.compute");
    ComputeShader bboxExtractionShader("assets/shaders/snd_parallel_attempt/cpu_cull/sph_bbox_ext.compute");
    ComputeShader bboxIntersectionShader("assets/shaders/snd_parallel_attempt/cpu_cull/sph_bbox_int.compute");

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

    std::vector<glm::vec4> spheres = getScene(sphere_count);
    const int actualSphereCount = static_cast<int>(spheres.size());
    std::vector<std::pair<glm::vec3, glm::vec3>> chkPoints = getCheckpoints(sphere_count, spheres);
    
    std::vector<SphereContainer> visibleSpheres(actualSphereCount);
    fillSpheresData(spheres, visibleSpheres);
    
    StorageBuffer sphereBuffer(GL_SHADER_STORAGE_BUFFER, actualSphereCount * sizeof(SphereContainer), 6, 
    visibleSpheres.data(), GL_STATIC_DRAW);
    
    StorageBuffer sphereBillboardBuffer(GL_SHADER_STORAGE_BUFFER, actualSphereCount * sizeof(SphereBillboard), 7,
        nullptr, GL_STATIC_DRAW);

    GLuint zero = 0;
    GLuint resetValue = 0;
    StorageBuffer visibleSpheresAtomicCounter(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint), 8, 
        &zero, GL_STATIC_DRAW);

    std::vector<GLuint> visibilityFrames(2 * actualSphereCount, 10);
    StorageBuffer visibilityFramesBuffer(GL_SHADER_STORAGE_BUFFER, 2 * actualSphereCount * sizeof(GLuint), 5, 
    visibilityFrames.data(), GL_DYNAMIC_COPY);
    

    Benchmark benchmark(camera_controller, chkPoints);
    Profiler profiler(window, "media/off/scnd_parallel/cpu_cull/frame_times.off", "media/off/scnd_parallel/cpu_cull/process_times.off", actualSphereCount, 0);
    
    window.setupSceneInfoGui("Scene Info", scene_data);
    window.setupCameraGui("Camera Info", &camera);
    window.setupInputInfoGui("General Input Info");
    window.setupBenchmarkInfoGui("Benchmark", &benchmark);

    // Calculating number of work groups (based on the number of threads and spheres)
    GLuint numGroupsX = (actualSphereCount + workGroupSizeXPerSphere - 1) / workGroupSizeXPerSphere;
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
            profiler.updateProfiler(benchmark.getCheckpointID(), visibleSpheresCount, notOccludedSpheresCount, 0, 0);
            
            if (window.getInput().isKeyDown(Key::T)) profiler.startSavingNextFrames(benchmark.getCheckpointID());

            canvas.downsampleCanvasDepth(downsampleWorkGroupSizeX, downsampleWorkGroupSizeY);
            
            canvas.cleanCanvasBuffers(workGroupSizeXPerPixel, workGroupSizeYPerPixel, camera.getFar());

            visibleSpheresAtomicCounter.bind();
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
            visibleSpheresAtomicCounter.unbind();

            Frustum frustum(camera);
            // bbox extraction shader
            bboxExtractionShader.use();
            cullSpheres(spheres, visibleSpheres, frustum, visibleSpheresCount);
            sphereBuffer.bind();
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, visibleSpheresCount * sizeof(SphereContainer), visibleSpheres.data());
            sphereBuffer.unbind();
            numGroupsX = (visibleSpheresCount + workGroupSizeXPerSphere - 1) / workGroupSizeXPerSphere;
            bboxExtractionShader.setInt("sphereCount", visibleSpheresCount); // visible sphere count given
            bboxExtractionShader.setUint("visibilityFrameBufferIndexOffset", 0);
            bboxExtractionShader.setVec2I("screenResolution", screenResolution);
            setCameraUniforms(bboxExtractionShader, camera);
            glDispatchCompute(numGroupsX, numGroupsY, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            
            // bbox intersection shader
            bboxIntersectionShader.use();
            bboxIntersectionShader.setFloat("far", camera.getFar());
            bboxIntersectionShader.setVec2I("screenResolution", screenResolution);
            setCameraUniforms(bboxIntersectionShader, camera);
            glDispatchCompute((SCR_WIDTH + workGroupSizeXPerPixel - 1) / workGroupSizeXPerPixel, 
                (SCR_HEIGHT + workGroupSizeYPerPixel - 1) / workGroupSizeYPerPixel, 
                1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

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
            //     visibleSpheresCount = 0;
            //     // looping on elements checking if drawn or not
            //     for (size_t i = 0; i < sphere_count; ++i)
            //         if (spheresData[i].wasDrawn[1]) 
            //             visibleSpheresCount++;
                
            //     // unmapping buffer once finished
            //     glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
            // } else {
            //     std::cerr << "Failed to map the buffer!" << std::endl;
            // }

            // sphereBuffer.unbind();
            
            ////// END:: checking drawn spheres ///////
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }


    return 0;
}