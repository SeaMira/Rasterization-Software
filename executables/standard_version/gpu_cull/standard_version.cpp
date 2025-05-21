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
#include "vis/shader_program.h"

using uint = unsigned int;

// Settings
int SCR_WIDTH = 1024;
int SCR_HEIGHT = 1024;
int sphere_count = 1000;
int cylinder_count = 1000;

std::string title = "Standard OpenGL Version: Spheres and Cylinders"; 

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

void mainWithOcclusionCulling(Camera& camera, AppOpenGL& window);
void mainWithoutOcclusionCulling(Camera& camera, AppOpenGL& window);

int main(int argc, char* argv[]) 
{
    AppOpenGL window { title, SCR_WIDTH, SCR_HEIGHT, shown };

    Camera camera(SCR_WIDTH, SCR_HEIGHT);
    camera.SetPosition(.0f, .0f, .0f);
    if (withOcclusionCulling)
        mainWithOcclusionCulling(camera, window);
    else
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

    // FBO
    const int mipLevels = 1 + static_cast<int>(std::floor(std::log2(std::max(SCR_WIDTH, SCR_HEIGHT))));
    Texture canvasImageTexture(GL_TEXTURE_2D, GL_RGBA8, SCR_WIDTH, SCR_HEIGHT, 0);
    Texture canvasPixelIdTexture(GL_TEXTURE_2D, GL_R32UI, SCR_WIDTH, SCR_HEIGHT, GL_RED_INTEGER, GL_UNSIGNED_INT, 0, nullptr);
    Texture canvasDepthTexture(GL_TEXTURE_2D, GL_DEPTH_COMPONENT32F, SCR_WIDTH, SCR_HEIGHT, 0, mipLevels);
    

    Framebuffer canvasFBO;
    canvasFBO.attachTexture(GL_COLOR_ATTACHMENT0, canvasImageTexture);
    canvasFBO.attachTexture(GL_COLOR_ATTACHMENT1, canvasPixelIdTexture);
    canvasFBO.attachTexture(GL_DEPTH_ATTACHMENT, canvasDepthTexture);

    GLenum drawBuffers[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    glBindFramebuffer(GL_FRAMEBUFFER, canvasFBO.getId());
    glDrawBuffers(2, drawBuffers);

    // FBO

    ComputeShader pixelCountShader("assets/shaders/standard_version/gpu_cull/pixel_count.compute");

    ComputeShader spheresCullingShader("assets/shaders/standard_version/gpu_cull/spheresFrusOccCulling.compute");
    ShaderProgram spheresShader("assets/shaders/standard_version/gpu_cull/spheres.vert", 
                                "assets/shaders/standard_version/gpu_cull/spheresOccCulling.frag",
                                "assets/shaders/standard_version/gpu_cull/spheresOccCulling.geom");
    spheresShader.linkProgram();
    
    ComputeShader cylindersCullingShader("assets/shaders/standard_version/gpu_cull/cylindersFrusOccCulling.compute");
    ShaderProgram cylindersShader("assets/shaders/standard_version/gpu_cull/cylinders.vert", 
                                "assets/shaders/standard_version/gpu_cull/cylindersOccCulling.frag",
                                "assets/shaders/standard_version/gpu_cull/cylindersOccCulling.geom");
    cylindersShader.linkProgram();

    std::vector<Sphere> spheres = getScene(sphere_count);
    sphere_count = spheres.size(); 
    
    std::vector<Cylinder> cylinders = getCylinderScene(cylinder_count);
    cylinder_count = cylinders.size(); 

    std::vector<std::pair<glm::vec3, glm::vec3>> chkPoints = getCheckpoints(sphere_count, spheres, cylinders);

    visibleSpheresCount = sphere_count;
    
    visibleCylindersCount = cylinder_count;

    std::cout << "Spheres buffer " << std::endl;
    StorageBuffer sphereBuffer(GL_SHADER_STORAGE_BUFFER, sphere_count * sizeof(Sphere), 1, 
        spheres.data(), GL_STATIC_DRAW);
    
    std::cout << "Sphere indexes buffer " << std::endl;
    std::vector<GLuint> visibleSpheres(sphere_count, 0);
    StorageBuffer visibleSpheresBuffer(GL_SHADER_STORAGE_BUFFER, sphere_count * sizeof(GLuint), 2, 
        visibleSpheres.data(), GL_STATIC_DRAW);

    GLuint zero = 0;
    GLuint resetValue = 0;
    StorageBuffer visibleEntitiesAtomicCounter(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint), 3, 
    &zero, GL_STATIC_DRAW);
        
    StorageBuffer cylinderBuffer(GL_SHADER_STORAGE_BUFFER, cylinder_count * sizeof(Cylinder), 4, 
        cylinders.data(), GL_STATIC_DRAW);

    std::vector<GLuint> visibleCylinders(cylinder_count, 0);
    StorageBuffer visibleCylindersBuffer(GL_SHADER_STORAGE_BUFFER, cylinder_count * sizeof(GLuint), 5, 
        visibleCylinders.data(), GL_STATIC_DRAW);


    ////////////// Buffers for occlusion culling
    int totalEntities = sphere_count + cylinder_count;
    std::vector<GLuint> visibilityFrames(2 * totalEntities, 10);
    StorageBuffer visibilityFramesBuffer(GL_SHADER_STORAGE_BUFFER, 2 * totalEntities * sizeof(GLuint), 6, 
        visibilityFrames.data(), GL_DYNAMIC_COPY);
    //////////////

    
    GLuint emptyVAO;
    glCreateVertexArrays(1, &emptyVAO);

    Benchmark benchmark(camera_controller, chkPoints);

    std::string base_path = "media/csv/standard_version/gpu_cull/"; 
    base_path += ((currentScene == SceneType::LOADED_SCENE) ? ("loaded_scene_" + scene_file) : "packed_scene");
    std::string frame_times_path = base_path + "_frame_times.csv";
    std::string process_times_path = base_path + "_process_times.csv";
    Profiler profiler(window, 
        frame_times_path, 
        process_times_path, sphere_count, 0, 0);

    window.setupSceneInfoGui("Scene Info", scene_data);
    window.setupCameraGui("Camera Info", &camera);
    window.setupInputInfoGui("General Input Info");
    window.setupBenchmarkInfoGui("Benchmark", &benchmark);

    GLuint numGroupsXSpheres = (sphere_count + workGroupSizeXPerSphere - 1) / workGroupSizeXPerSphere;
    GLuint numGroupsXCylinders = (cylinder_count + workGroupSizeXPerCylinder - 1) / workGroupSizeXPerCylinder;
    GLuint numGroupsY = 1;

    glm::ivec2 screenResolution(SCR_WIDTH, SCR_HEIGHT);

    try
    {
       bool isRunning = true;
        while ( isRunning )
        {
            // 1. Binding framebuffer to draw on.
            glBindFramebuffer(GL_FRAMEBUFFER, canvasFBO.getId());

            // 2. Setting drawing area.
            glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);

            // 3. Choose background color.
            glClearColor(1.0f, 1.0f, 0.0f, 1.0f);   // amarillo, por ejemplo

            // 4. Crlean depth and color buffers.
            glGenerateTextureMipmap(canvasDepthTexture.getId());
            
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            
            glEnable(GL_DEPTH_TEST);
            ////////////////
            
            camera_controller.cameraUpdate();
            benchmark.update();
            profiler.updateProfiler(benchmark.getCheckpointID(), visibleSpheresCount, notOccludedSpheresCount, 0, 0);

            if (window.getInput().isKeyDown(Key::T)) profiler.startSavingNextFrames(benchmark.getCheckpointID());
            
            Frustum frustum(camera);

            /// Spheres drawing
            glBindTextureUnit(0, canvasDepthTexture.getId());
            spheresCullingShader.use();
            spheresCullingShader.setInt("sphereCount", sphere_count);
            spheresCullingShader.setInt("mipmapLevels", mipLevels);
            spheresCullingShader.setUint("indexOffset", 0);
            spheresCullingShader.setVec2I("screenResolution", screenResolution);
            setCameraUniforms(spheresCullingShader, camera);
            setFrustumUniforms(spheresCullingShader, frustum);
            glDispatchCompute(numGroupsXSpheres, numGroupsY, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
            glBindTextureUnit(0, 0);
            
            visibleEntitiesAtomicCounter.bind();
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &visibleSpheresCount);
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
            visibleEntitiesAtomicCounter.unbind();

            spheresShader.use();
            spheresShader.setUint("indexOffset", 0);
            spheresShader.setVec2I("screenResolution", screenResolution);
            setCameraUniforms(spheresShader, camera);
            glBindVertexArray(emptyVAO);
            glDrawArrays(GL_POINTS, 0, visibleSpheresCount);
            
            /// Cylinder drawing
            glBindTextureUnit(0, canvasDepthTexture.getId());
            cylindersCullingShader.use();
            cylindersCullingShader.setInt("cylinderCount", cylinder_count);
            cylindersCullingShader.setInt("mipmapLevels", mipLevels);
            cylindersCullingShader.setUint("indexOffset", sphere_count);
            cylindersCullingShader.setVec2I("screenResolution", screenResolution);
            setCameraUniforms(cylindersCullingShader, camera);
            setFrustumUniforms(cylindersCullingShader, frustum);
            glDispatchCompute(numGroupsXCylinders, numGroupsY, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
            glBindTextureUnit(0, 0);
            
            visibleEntitiesAtomicCounter.bind();
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &visibleCylindersCount);
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
            visibleEntitiesAtomicCounter.unbind();

            cylindersShader.use();
            cylindersShader.setUint("indexOffset", sphere_count);
            cylindersShader.setVec2I("screenResolution", screenResolution);
            setCameraUniforms(cylindersShader, camera);
            glBindVertexArray(emptyVAO);
            glDrawArrays(GL_POINTS, 0, visibleCylindersCount);


            canvasPixelIdTexture.bindImage(GL_READ_ONLY, GL_R32UI);
            pixelCountShader.use();
            pixelCountShader.setVec2I("screenResolution", screenResolution);
            glDispatchCompute((SCR_WIDTH + workGroupSizeXPerPixel - 1) / workGroupSizeXPerPixel, 
                (SCR_HEIGHT + workGroupSizeYPerPixel - 1) / workGroupSizeYPerPixel, 
                1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            canvasPixelIdTexture.unbindImage(GL_READ_ONLY, GL_R32UI);
            ////////////////
            glBindFramebuffer(GL_READ_FRAMEBUFFER, canvasFBO.getId());
            isRunning = window.update();
        } 
    } catch (const std::exception& e)
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
        {"Spheres On Frustum", &visibleSpheresCount},
        {"Drawn spheres", &notOccludedSpheresCount},
        {"Cylinder count", &cylinder_count},
        {"Cylinders On Frustum", &visibleCylindersCount},
        {"Drawn cylinders", &notOccludedCylindersCount}
    };

    CameraController camera_controller(window, camera);

    // FBO
    const int mipLevels = 1 + static_cast<int>(std::floor(std::log2(std::max(SCR_WIDTH, SCR_HEIGHT))));
    Texture canvasImageTexture(GL_TEXTURE_2D, GL_RGBA8, SCR_WIDTH, SCR_HEIGHT, 0);
    Texture canvasDepthTexture(GL_TEXTURE_2D, GL_DEPTH_COMPONENT32F, SCR_WIDTH, SCR_HEIGHT, 0, mipLevels);
    

    Framebuffer canvasFBO;
    canvasFBO.attachTexture(GL_COLOR_ATTACHMENT0, canvasImageTexture);
    canvasFBO.attachTexture(GL_DEPTH_ATTACHMENT, canvasDepthTexture);

    // FBO

    ComputeShader spheresCullingShader("assets/shaders/standard_version/gpu_cull/spheresFrusCulling.compute");
    ShaderProgram spheresShader("assets/shaders/standard_version/gpu_cull/spheres.vert", 
                                "assets/shaders/standard_version/gpu_cull/spheres.frag",
                                "assets/shaders/standard_version/gpu_cull/spheres.geom");
    spheresShader.linkProgram();
    
    ComputeShader cylindersCullingShader("assets/shaders/standard_version/gpu_cull/cylindersFrusCulling.compute");
    ShaderProgram cylindersShader("assets/shaders/standard_version/gpu_cull/cylinders.vert", 
                                "assets/shaders/standard_version/gpu_cull/cylinders.frag",
                                "assets/shaders/standard_version/gpu_cull/cylinders.geom");
    cylindersShader.linkProgram();

    std::vector<Sphere> spheres = getScene(sphere_count);
    sphere_count = spheres.size(); 
    
    std::vector<Cylinder> cylinders = getCylinderScene(cylinder_count);
    cylinder_count = cylinders.size(); 

    std::vector<std::pair<glm::vec3, glm::vec3>> chkPoints = getCheckpoints(sphere_count, spheres, cylinders);

    visibleSpheresCount = sphere_count;
    
    visibleCylindersCount = cylinder_count;

    std::cout << "Spheres buffer " << std::endl;
    StorageBuffer sphereBuffer(GL_SHADER_STORAGE_BUFFER, sphere_count * sizeof(Sphere), 1, 
        spheres.data(), GL_STATIC_DRAW);
    
    std::cout << "Sphere indexes buffer " << std::endl;
    std::vector<GLuint> visibleSpheres(sphere_count, 0);
    StorageBuffer visibleSpheresBuffer(GL_SHADER_STORAGE_BUFFER, sphere_count * sizeof(GLuint), 2, 
        visibleSpheres.data(), GL_STATIC_DRAW);

    GLuint zero = 0;
    GLuint resetValue = 0;
    StorageBuffer visibleEntitiesAtomicCounter(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint), 3, 
    &zero, GL_STATIC_DRAW);
        
    StorageBuffer cylinderBuffer(GL_SHADER_STORAGE_BUFFER, cylinder_count * sizeof(Cylinder), 4, 
        cylinders.data(), GL_STATIC_DRAW);

    std::vector<GLuint> visibleCylinders(cylinder_count, 0);
    StorageBuffer visibleCylindersBuffer(GL_SHADER_STORAGE_BUFFER, cylinder_count * sizeof(GLuint), 5, 
        visibleCylinders.data(), GL_STATIC_DRAW);

    GLuint emptyVAO;
    glCreateVertexArrays(1, &emptyVAO);

    Benchmark benchmark(camera_controller, chkPoints);

    std::string base_path = "media/csv/standard_version/gpu_cull/"; 
    base_path += ((currentScene == SceneType::LOADED_SCENE) ? ("loaded_scene_" + scene_file) : "packed_scene");
    std::string frame_times_path = base_path + "_frame_times.csv";
    std::string process_times_path = base_path + "_process_times.csv";
    Profiler profiler(window, 
        frame_times_path, 
        process_times_path, sphere_count, 0, 0);

    window.setupSceneInfoGui("Scene Info", scene_data);
    window.setupCameraGui("Camera Info", &camera);
    window.setupInputInfoGui("General Input Info");
    window.setupBenchmarkInfoGui("Benchmark", &benchmark);

    GLuint numGroupsXSpheres = (sphere_count + workGroupSizeXPerSphere - 1) / workGroupSizeXPerSphere;
    GLuint numGroupsXCylinders = (cylinder_count + workGroupSizeXPerCylinder - 1) / workGroupSizeXPerCylinder;
    GLuint numGroupsY = 1;

    glm::ivec2 screenResolution(SCR_WIDTH, SCR_HEIGHT);

    try
    {
       bool isRunning = true;
        while ( isRunning )
        {
            // 1. Ligar el framebuffer en el que vas a dibujar
            glBindFramebuffer(GL_FRAMEBUFFER, canvasFBO.getId());

            // 2. Ajustar el área de dibujo
            glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);

            // 3. Elegir el color de fondo
            //    (RGBA en el rango 0.0 – 1.0)
            glClearColor(1.0f, 1.0f, 0.0f, 1.0f);   // amarillo, por ejemplo

            // 4. Limpiar color y profundidad
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            
            glGenerateTextureMipmap(canvasDepthTexture.getId());
            
            glEnable(GL_DEPTH_TEST);
            ////////////////
            
            camera_controller.cameraUpdate();
            benchmark.update();
            profiler.updateProfiler(benchmark.getCheckpointID(), visibleSpheresCount, notOccludedSpheresCount, 0, 0);

            if (window.getInput().isKeyDown(Key::T)) profiler.startSavingNextFrames(benchmark.getCheckpointID());
            
            Frustum frustum(camera);

            /// Spheres drawing
            spheresCullingShader.use();
            spheresCullingShader.setInt("sphereCount", sphere_count);
            setFrustumUniforms(spheresCullingShader, frustum);
            glDispatchCompute(numGroupsXSpheres, numGroupsY, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
            
            visibleEntitiesAtomicCounter.bind();
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &visibleSpheresCount);
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
            visibleEntitiesAtomicCounter.unbind();

            spheresShader.use();
            spheresShader.setVec2I("screenResolution", screenResolution);
            setCameraUniforms(spheresShader, camera);
            glBindVertexArray(emptyVAO);
            glDrawArrays(GL_POINTS, 0, visibleSpheresCount);
            
            /// Cylinder drawing
            cylindersCullingShader.use();
            cylindersCullingShader.setInt("cylinderCount", cylinder_count);
            setFrustumUniforms(cylindersCullingShader, frustum);
            glDispatchCompute(numGroupsXCylinders, numGroupsY, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
            
            visibleEntitiesAtomicCounter.bind();
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &visibleCylindersCount);
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
            visibleEntitiesAtomicCounter.unbind();

            cylindersShader.use();
            cylindersShader.setVec2I("screenResolution", screenResolution);
            setCameraUniforms(cylindersShader, camera);
            glBindVertexArray(emptyVAO);
            glDrawArrays(GL_POINTS, 0, visibleCylindersCount);

            ////////////////
            glBindFramebuffer(GL_READ_FRAMEBUFFER, canvasFBO.getId());
            isRunning = window.update();
        } 
    } catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return;
    }
    
}