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
int sphere_count = 126;

std::string title = "First Parallel Version"; 

bool shown = true;

GLuint workGroupSizeXPerPixel = 16;  // Deifining threads-per-group (X)
GLuint workGroupSizeYPerPixel = 16;  // Deifining threads-per-group (Y)

GLuint workGroupSizeXPerSphere = 256;  // Deifining threads-per-sphere (X)

int frustumSpheres = 0;
int drawnSpheres = 0;

// downsample settings
int downsampleLevel = 4;
int downsampleWorkGroupSizeX = 16/downsampleLevel;
int downsampleWorkGroupSizeY = 16/downsampleLevel;

int main(int argc, char* argv[]) 
{

    std::unordered_map<std::string, int*> scene_data = {
        {"Screen width", &SCR_WIDTH},
        {"Screen height", &SCR_HEIGHT},
        {"Sphere count", &sphere_count},
        {"Spheres On Frustum", &frustumSpheres},
        {"Drawn spheres", &drawnSpheres}
    };

    AppOpenGL window { title, SCR_WIDTH, SCR_HEIGHT, shown };

    Camera camera(SCR_WIDTH, SCR_HEIGHT);
    camera.SetPosition(.0f, .0f, .0f);
    CameraController camera_controller(window, camera);
    
    #if CPU_FRUSTUM_CULLING
        ComputeShader computeShader("assets/shaders/fst_parallel_attempt/sphere.compute");
    #else
        ComputeShader computeShader("assets/shaders/fst_parallel_attempt/sphere_culling.compute");
    #endif
    ComputeShader cleaningComputeShader("assets/shaders/fst_parallel_attempt/set_to_black.compute");
    ComputeShader hizPyramidComputeShader("assets/shaders/fst_parallel_attempt/mipmap_gen.compute");
    ComputeShader pixelCountComputeShader("assets/shaders/fst_parallel_attempt/pixel_count.compute");

    Canvas canvas(GL_TEXTURE_2D, GL_RGBA8, SCR_WIDTH, SCR_HEIGHT);
    canvas.setFBO(GL_COLOR_ATTACHMENT0);
    
    if (!canvas.getFramebuffer().isComplete()) {
        throw std::runtime_error("Error: Incomplete Framebuffer.");
    }
    
    std::vector<glm::vec4> spheres = getScene(sphere_count);
    sphere_count = spheres.size(); 
    std::vector<std::pair<glm::vec3, glm::vec3>> chkPoints = getCheckpoints(sphere_count, spheres);
    
    std::vector<SphereContainer> visibleSpheres(spheres.size());    
    #if !CPU_FRUSTUM_CULLING
        frustumSpheres = sphere_count;
        fillSpheresData(spheres, visibleSpheres);
    #endif
    
    StorageBuffer sphereBuffer(GL_SHADER_STORAGE_BUFFER);
    sphereBuffer.generateBufferData(spheres.size() * sizeof(SphereContainer), 1, 
        visibleSpheres.data(), GL_STATIC_DRAW);
    sphereBuffer.unbind();
    
    Texture depthTexture(GL_TEXTURE_2D, GL_R32F, SCR_WIDTH, SCR_HEIGHT, 3);
    Texture downsampledDepthTexture(GL_TEXTURE_2D, GL_R32F, SCR_WIDTH/(1 << downsampleLevel), SCR_HEIGHT/(1 << downsampleLevel), 4);
    // Framebuffer downsampleDepthFBO;
    // downsampleDepthFBO.attachTexture(GL_COLOR_ATTACHMENT0, downsampledDepthTexture);
    
    std::vector<GLuint> visibilityFrames(2 * spheres.size(), 10);
    StorageBuffer visibilityFramesBuffer(GL_SHADER_STORAGE_BUFFER);
    visibilityFramesBuffer.generateBufferData(2 * spheres.size() * sizeof(GLuint), 5, 
    visibilityFrames.data(), GL_DYNAMIC_COPY);
    visibilityFramesBuffer.unbind();
    
    std::vector<GLuint> pixelCountFrames(SCR_WIDTH*SCR_HEIGHT, 0);
    StorageBuffer pixelCountFramesBuffer(GL_SHADER_STORAGE_BUFFER);
    pixelCountFramesBuffer.generateBufferData(SCR_WIDTH*SCR_HEIGHT * sizeof(GLuint), 6, 
    pixelCountFrames.data(), GL_DYNAMIC_COPY);
    pixelCountFramesBuffer.unbind();
    
    StorageBuffer depthBuffer(GL_SHADER_STORAGE_BUFFER);
    depthBuffer.generateBufferData(SCR_HEIGHT * SCR_WIDTH * sizeof(uint), 7, 
        nullptr, GL_STATIC_DRAW);
    depthBuffer.unbind();

    Benchmark benchmark(camera_controller, chkPoints);
    
    #if CPU_FRUSTUM_CULLING
        Profiler profiler(window, "media/off/fst_parallel/cpu_frame_times.off", "media/off/fst_parallel/cpu_process_times.off", sphere_count, 0);
    #else
        Profiler profiler(window, "media/off/fst_parallel/gpu_frame_times.off", "media/off/fst_parallel/gpu_process_times.off", sphere_count, 0);
    #endif
    window.setupSceneInfoGui("Scene Info", scene_data);
    window.setupCameraGui("Camera Info", &camera);
    window.setupInputInfoGui("General Input Info");
    window.setupBenchmarkInfoGui("Benchmark", &benchmark);
    
    // Calculating number of work groups (based on the number of threads and spheres)
    GLuint numGroupsX = (sphere_count + workGroupSizeXPerSphere - 1) / workGroupSizeXPerSphere;
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
            profiler.updateProfiler(benchmark.getCheckpointID(), frustumSpheres, drawnSpheres, 0, 0);

            if (window.getInput().isKeyDown(Key::T)) profiler.startSavingNextFrames(benchmark.getCheckpointID());

            hizPyramidComputeShader.use();
            downsampledDepthTexture.bindImage(GL_READ_WRITE, GL_R32F);
            depthTexture.unbindImage(GL_READ_WRITE, GL_R32F);
            depthTexture.bind();
            glm::vec2 utexelDimensions = glm::vec2( 1.0f / (float)SCR_WIDTH, 1.0f / (float)SCR_HEIGHT );
            hizPyramidComputeShader.setInt("depthTextureSampler", 2);
            hizPyramidComputeShader.setVec2("utexelDimensions", utexelDimensions);
            glDispatchCompute((SCR_WIDTH + downsampleWorkGroupSizeX - 1) / downsampleWorkGroupSizeX, (SCR_HEIGHT + downsampleWorkGroupSizeY - 1) / downsampleWorkGroupSizeY, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

            depthTexture.bindImage(GL_READ_WRITE, GL_R32F);

            cleaningComputeShader.use();
            cleaningComputeShader.setFloat("far", camera.getFar());
            cleaningComputeShader.setVec2I("screenResolution", screenResolution);
            glDispatchCompute((SCR_WIDTH + workGroupSizeXPerPixel - 1) / workGroupSizeXPerPixel, 
                (SCR_HEIGHT + workGroupSizeYPerPixel - 1) / workGroupSizeYPerPixel, 
                1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);


            downsampledDepthTexture.unbindImage(GL_READ_WRITE, GL_R32F);
            downsampledDepthTexture.bind();

            computeShader.use();
            Frustum frustum(camera);
            #if CPU_FRUSTUM_CULLING
                cullSpheres(spheres, visibleSpheres, frustum, frustumSpheres);
                sphereBuffer.bind();
                glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, frustumSpheres * sizeof(SphereContainer), visibleSpheres.data());
                sphereBuffer.unbind();
                numGroupsX = (frustumSpheres + workGroupSizeXPerSphere - 1) / workGroupSizeXPerSphere;
                
                computeShader.setInt("sphereCount", frustumSpheres); // visible sphere count given
                #else
                computeShader.setInt("sphereCount", sphere_count);
                computeShader.setVec4("frustumTopFace", glm::vec4(frustum.topFace.normal, frustum.topFace.distance));
                computeShader.setVec4("frustumBottomFace", glm::vec4(frustum.bottomFace.normal, frustum.bottomFace.distance));
                computeShader.setVec4("frustumRightFace", glm::vec4(frustum.rightFace.normal, frustum.rightFace.distance));
                computeShader.setVec4("frustumLeftFace", glm::vec4(frustum.leftFace.normal, frustum.leftFace.distance));
                computeShader.setVec4("frustumFarFace", glm::vec4(frustum.farFace.normal, frustum.farFace.distance));
                computeShader.setVec4("frustumNearFace", glm::vec4(frustum.nearFace.normal, frustum.nearFace.distance));
            #endif
            computeShader.setVec2I("screenResolution", screenResolution);
            computeShader.setMat4("proj", camera.getProjection());
            computeShader.setMat4("view", camera.getView());
            computeShader.setVec3("up", camera.getUp());
            computeShader.setVec3("front", camera.getFront());
            computeShader.setVec3("right", camera.getRight());
            computeShader.setVec3("cameraPos", camera.getPosition());
            computeShader.setFloat("fov", camera.getFov()); // visible sphere count given

            glDispatchCompute(numGroupsX, numGroupsY, 1);
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
            //     drawnSpheres = 0;
            //     // looping on elements checking if drawn or not
            //     for (size_t i = 0; i < frustumSpheres; ++i)
            //         if (spheresData[i].wasDrawn[0]) 
            //             drawnSpheres++;
                
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

            if (window.getInput().isKeyDown(Key::F)) fbo1o2 = !fbo1o2; 

        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }


    return 0;
}