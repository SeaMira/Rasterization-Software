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
int sphere_count = 0;
int cylinder_count = 2000000;

std::string title = "First Parallel Version: Spheres and Cylinders"; 

bool shown = true;

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

void setCameraUniforms(ComputeShader& shader, Camera& camera);
void setFrustumUniforms(ComputeShader& shader, Frustum& frustum);

int main(int argc, char* argv[]) 
{

    std::unordered_map<std::string, int*> scene_data = {
        {"Screen width", &SCR_WIDTH},
        {"Screen height", &SCR_HEIGHT},
        {"Sphere count", &sphere_count},
        {"Cylinder count", &cylinder_count},
        {"Spheres On Frustum", &visibleSpheresCount},
        {"Drawn spheres", &notOccludedSpheresCount}
    };

    AppOpenGL window { title, SCR_WIDTH, SCR_HEIGHT, shown };

    Camera camera(SCR_WIDTH, SCR_HEIGHT);
    camera.SetPosition(.0f, .0f, .0f);
    CameraController camera_controller(window, camera);
    
    #if CPU_FRUSTUM_CULLING
        ComputeShader spheresShader("assets/shaders/fst_parallel_attempt/sphere.compute");
        ComputeShader cylinderShader("assets/shaders/fst_parallel_attempt/cylinder.compute");
    #else
        ComputeShader spheresShader("assets/shaders/fst_parallel_attempt/sphere_culling.compute");
        ComputeShader cylinderShader("assets/shaders/fst_parallel_attempt/cylinder_culling.compute");
    #endif
    ComputeShader cleaningComputeShader("assets/shaders/fst_parallel_attempt/set_to_black.compute");
    ComputeShader hizPyramidComputeShader("assets/shaders/fst_parallel_attempt/mipmap_gen.compute");
    ComputeShader pixelCountComputeShader("assets/shaders/fst_parallel_attempt/pixel_count.compute");

    Canvas canvas(GL_TEXTURE_2D, GL_RGBA8, SCR_WIDTH, SCR_HEIGHT);
    canvas.setFBO(GL_COLOR_ATTACHMENT0);
    
    if (!canvas.getFramebuffer().isComplete()) {
        throw std::runtime_error("Error: Incomplete Framebuffer.");
    }
    
    std::vector<Sphere> spheres = getScene(sphere_count);
    sphere_count = spheres.size(); 

    std::vector<Cylinder> cylinders = getCylinderScene(cylinder_count);

    cylinder_count = cylinders.size(); 

    std::vector<std::pair<glm::vec3, glm::vec3>> chkPoints = getCheckpoints(sphere_count, spheres);
    
    std::vector<SphereContainer> visibleSpheres(sphere_count);    
    std::vector<CylinderContainer> visibleCylinders(cylinder_count);    
    #if !CPU_FRUSTUM_CULLING
        visibleSpheresCount = sphere_count;
        fillSpheresData(spheres, visibleSpheres);
        
        visibleCylindersCount = cylinder_count;
        fillCylindersData(cylinders, visibleCylinders);
    #endif
    
    StorageBuffer sphereBuffer(GL_SHADER_STORAGE_BUFFER);
    sphereBuffer.generateBufferData(sphere_count * sizeof(SphereContainer), 1, 
        visibleSpheres.data(), GL_STATIC_DRAW);
    sphereBuffer.unbind();
    
    StorageBuffer cylinderBuffer(GL_SHADER_STORAGE_BUFFER);
    cylinderBuffer.generateBufferData(cylinder_count * sizeof(CylinderContainer), 2, 
        visibleCylinders.data(), GL_STATIC_DRAW);
    cylinderBuffer.unbind();
    
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
    Profiler profiler(window, "media/off/fst_parallel/frame_times.off", "media/off/fst_parallel/process_times.off");
    
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
            profiler.updateProfiler();

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

            Frustum frustum(camera);

            spheresShader.use();
            #if CPU_FRUSTUM_CULLING
                cullSpheres(spheres, visibleSpheres, frustum, visibleSpheresCount);
                sphereBuffer.bind();
                glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, visibleSpheresCount * sizeof(SphereContainer), visibleSpheres.data());
                sphereBuffer.unbind();
                numGroupsXSpheres = (visibleSpheresCount + workGroupSizeXPerSphere - 1) / workGroupSizeXPerSphere;
                
                spheresShader.setInt("sphereCount", visibleSpheresCount); // visible sphere count given
            
            #else
                spheresShader.setInt("sphereCount", sphere_count);
                setFrustumUniforms(spheresShader, frustum);

            #endif
            spheresShader.setVec2I("screenResolution", screenResolution);
            setCameraUniforms(spheresShader, camera);

            glDispatchCompute(numGroupsXSpheres, numGroupsY, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
            
            cylinderShader.use();
            #if CPU_FRUSTUM_CULLING
                cullCylinders(cylinders, visibleCylinders, frustum, visibleCylindersCount);
                cylinderBuffer.bind();
                glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, visibleCylindersCount * sizeof(CylinderContainer), visibleCylinders.data());
                cylinderBuffer.unbind();
                numGroupsXCylinders = (visibleCylindersCount + workGroupSizeXPerSphere - 1) / workGroupSizeXPerSphere;
                
                cylinderShader.setInt("cylinderCount", visibleCylindersCount); // visible cylinder count given

            #else
                cylinderShader.setInt("cylinderCount", cylinder_count);
                setFrustumUniforms(cylinderShader, frustum);

            #endif
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

void setCameraUniforms(ComputeShader& shader, Camera& camera)
{
    shader.setMat4("proj", camera.getProjection());
    shader.setMat4("view", camera.getView());
    shader.setVec3("up", camera.getUp());
    shader.setVec3("front", camera.getFront());
    shader.setVec3("right", camera.getRight());
    shader.setVec3("cameraPos", camera.getPosition());
    shader.setFloat("fov", camera.getFov());
}

void setFrustumUniforms(ComputeShader& shader, Frustum& frustum)
{
    shader.setVec4("frustumTopFace", glm::vec4(frustum.topFace.normal, frustum.topFace.distance));
    shader.setVec4("frustumBottomFace", glm::vec4(frustum.bottomFace.normal, frustum.bottomFace.distance));
    shader.setVec4("frustumRightFace", glm::vec4(frustum.rightFace.normal, frustum.rightFace.distance));
    shader.setVec4("frustumLeftFace", glm::vec4(frustum.leftFace.normal, frustum.leftFace.distance));
    shader.setVec4("frustumFarFace", glm::vec4(frustum.farFace.normal, frustum.farFace.distance));
    shader.setVec4("frustumNearFace", glm::vec4(frustum.nearFace.normal, frustum.nearFace.distance));
}