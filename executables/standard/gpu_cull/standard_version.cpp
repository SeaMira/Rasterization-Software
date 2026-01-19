#include <iostream>
#include <fstream>
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
#include "vis/shader_program.h"

using uint = unsigned int;

// Settings
int SCR_WIDTH = 1024;
int SCR_HEIGHT = 1024;
int sphere_count = 552008;
int cylinder_count = 519767;

std::string title = "Standard OpenGL Version: Spheres and Cylinders - GPU Frustum Culling"; 

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

int downsampleLevel = 4;
int downsampleWorkGroupSizeX = 16/downsampleLevel;
int downsampleWorkGroupSizeY = 16/downsampleLevel;

double timerDuration = 48.0;

void loadConfiguration() {
    try {
        SceneSettings settings = SceneConfigLoader::loadDefault();
        SCR_WIDTH = settings.screenWidth;
        SCR_HEIGHT = settings.screenHeight;
        sphere_count = settings.sphereCount;
        cylinder_count = settings.cylinderCount;
        withOcclusionCulling = settings.withOcclusionCulling;
        downsampleLevel = settings.downsampleLevel;
        downsampleWorkGroupSizeX = 16 / downsampleLevel;
        downsampleWorkGroupSizeY = 16 / downsampleLevel;
        workGroupSizeXPerPixel = settings.workGroupSizePerPixelX;
        workGroupSizeYPerPixel = settings.workGroupSizePerPixelY;
        workGroupSizeXPerSphere = settings.workGroupSizePerSphere;
        workGroupSizeXPerCylinder = settings.workGroupSizePerCylinder;
        timerDuration = settings.timerDuration;
        std::cout << "Configuration loaded successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load configuration: " << e.what() << std::endl;
        std::cerr << "Using default values." << std::endl;
    }
}

void mainWithOcclusionCulling(Camera& camera, AppOpenGL& window);
void mainWithoutOcclusionCulling(Camera& camera, AppOpenGL& window);

std::chrono::steady_clock::time_point startTime = std::chrono::high_resolution_clock::now();

int main(int argc, char* argv[]) 
{
    loadConfiguration();
    
    startTime = std::chrono::high_resolution_clock::now();
    AppOpenGL window { title + (withOcclusionCulling ? " - With Occlusion Culling" : " - Without Occlusion Culling"), SCR_WIDTH, SCR_HEIGHT, shown };

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
    Texture canvasImageTexture(GL_TEXTURE_2D, GL_RGBA8, SCR_WIDTH, SCR_HEIGHT, 0);
    Texture canvasPixelIdTexture(GL_TEXTURE_2D, GL_R32UI, SCR_WIDTH, SCR_HEIGHT, GL_RED_INTEGER, GL_UNSIGNED_INT, 0, nullptr);
    Texture canvasDepthTexture(GL_TEXTURE_2D, GL_DEPTH_COMPONENT32F, SCR_WIDTH, SCR_HEIGHT, 0);
    Texture downsSampledDepthTexture(GL_TEXTURE_2D, GL_R32F, SCR_WIDTH/(1 << downsampleLevel), SCR_HEIGHT/(1 << downsampleLevel), 1, downsampleLevel);
    

    Framebuffer canvasFBO;
    canvasFBO.attachTexture(GL_COLOR_ATTACHMENT0, canvasImageTexture);
    canvasFBO.attachTexture(GL_COLOR_ATTACHMENT1, canvasPixelIdTexture);
    canvasFBO.attachTexture(GL_DEPTH_ATTACHMENT, canvasDepthTexture);

    GLenum drawBuffers[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    glBindFramebuffer(GL_FRAMEBUFFER, canvasFBO.getId());
    glDrawBuffers(2, drawBuffers);

    // FBO

    ComputeShader pixelCountShader("assets/shaders/standard_version/gpu_cull/pixel_count.compute", "Pixel Count Shader");
    ComputeShader hizPyramidComputeShader("assets/shaders/standard_version/gpu_cull/mipmap_gen.compute", "Hiz Pyramid Shader");

    ComputeShader spheresCullingShader("assets/shaders/standard_version/gpu_cull/spheresFrusOccCulling.compute", "Spheres Culling Shader");
    ShaderProgram spheresShader("assets/shaders/standard_version/gpu_cull/spheres.vert", 
                                "assets/shaders/standard_version/gpu_cull/spheresOccCulling.frag",
                                "assets/shaders/standard_version/gpu_cull/spheresOccCulling.geom",
                                "Spheres With Occ Culling Program",
                                "Spheres Vertex Shader",
                                "Spheres Fragment Shader",
                                "Spheres Geometry Shader");
    spheresShader.linkProgram();
    
    ComputeShader cylindersCullingShader("assets/shaders/standard_version/gpu_cull/cylindersFrusOccCulling.compute", "Cylinders Culling Shader");
    ShaderProgram cylindersShader("assets/shaders/standard_version/gpu_cull/cylinders.vert", 
                                "assets/shaders/standard_version/gpu_cull/cylindersOccCulling.frag",
                                "assets/shaders/standard_version/gpu_cull/cylindersOccCulling.geom",
                                "Cylinders Program",
                                "Cylinders Vertex Shader",
                                "Cylinders Fragment Shader",
                                "Cylinders Geometry Shader");   
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

    StorageBuffer frustumEntitiesAtomicCounter(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint), 7, 
    &zero, GL_STATIC_DRAW);
    
    GLuint emptyVAO;
    glCreateVertexArrays(1, &emptyVAO);

    Benchmark benchmark(camera_controller, chkPoints);

    std::string base_path = "media/csv/standard_version/gpu_cull/occ_"; 
    base_path += ((currentScene == SceneType::LOADED_SCENE) ? ("loaded_scene_" + scene_file) : "packed_scene");
    std::string frame_times_path = base_path + "_frame_times.csv";
    std::string process_times_path = base_path + "_process_times.csv";
    Profiler profiler(window, 
        frame_times_path, 
        process_times_path, sphere_count, cylinder_count, downsampleLevel, timerDuration);

    window.setupSceneInfoGui("Scene Info", scene_data);
    window.setupCameraGui("Camera Info", &camera);
    window.setupInputInfoGui("General Input Info");
    window.setupBenchmarkInfoGui("Benchmark", &benchmark);

    GLuint numGroupsXSpheres = (sphere_count + workGroupSizeXPerSphere - 1) / workGroupSizeXPerSphere;
    GLuint numGroupsXCylinders = (cylinder_count + workGroupSizeXPerCylinder - 1) / workGroupSizeXPerCylinder;
    GLuint numGroupsY = 1;

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
            // 1. Binding framebuffer to draw on.
            glBindFramebuffer(GL_FRAMEBUFFER, canvasFBO.getId());

            // 2. Setting drawing area.
            glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);

            // 3. Choose background color.
            glClearColor(1.0f, 1.0f, 0.0f, 1.0f);   // amarillo, por ejemplo

            // 4. Crlean depth and color buffers.
            canvasDepthTexture.bind();
            downsSampledDepthTexture.bindImage(GL_READ_WRITE, GL_R32F);

            hizPyramidComputeShader.use();
            glm::vec2 utexelDimensions = glm::vec2( 1.0f / (float)SCR_WIDTH, 1.0f / (float)SCR_HEIGHT );
            hizPyramidComputeShader.setVec2("utexelDimensions", utexelDimensions);
            glDispatchCompute((SCR_WIDTH + downsampleWorkGroupSizeX - 1) / downsampleWorkGroupSizeX, 
            (SCR_HEIGHT + downsampleWorkGroupSizeY - 1) / downsampleWorkGroupSizeY, 
            1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
            canvasDepthTexture.unbind();
            downsSampledDepthTexture.unbindImage(GL_READ_WRITE, GL_R32F);



            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            
            glEnable(GL_DEPTH_TEST);
            ////////////////
            
            camera_controller.cameraUpdate();
            benchmark.update();
            profiler.updateProfiler(benchmark.getCheckpointID(), visibleSpheresCount, notOccludedSpheresCount, visibleCylindersCount, notOccludedCylindersCount);

            if (window.getInput().isKeyDown(Key::T)) profiler.startSavingNextFrames(benchmark.getCheckpointID());
            
            Frustum frustum(camera);

            glBindTextureUnit(0, downsSampledDepthTexture.getId());
            
            /// Spheres drawing
            spheresCullingShader.use();
            spheresCullingShader.setInt("sphereCount", sphere_count);
            spheresCullingShader.setUint("indexOffset", 0);
            spheresCullingShader.setVec2I("screenResolution", screenResolution);
            #if BENCHMARKING 

                spheresCullingShader.setInt("benchmark", 1); 
            #else
                spheresCullingShader.setInt("benchmark", 0);
            #endif
            setCameraUniforms(spheresCullingShader, camera);
            setFrustumUniforms(spheresCullingShader, frustum);
            glDispatchCompute(numGroupsXSpheres, numGroupsY, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

            #if BENCHMARKING 

                frustumEntitiesAtomicCounter.bind();
                glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &visibleSpheresCount);
                glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
                frustumEntitiesAtomicCounter.unbind(); 
                
                visibleEntitiesAtomicCounter.bind();
                glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &notOccludedSpheresCount);
                glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
                visibleEntitiesAtomicCounter.unbind();
            #else
            
                visibleEntitiesAtomicCounter.bind();
                glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &notOccludedSpheresCount);
                glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
                visibleSpheresCount = notOccludedSpheresCount;
                visibleEntitiesAtomicCounter.unbind();
            #endif
            

            spheresShader.use();
            spheresShader.setUint("indexOffset", 0);
            spheresShader.setVec2I("screenResolution", screenResolution);
            setCameraUniforms(spheresShader, camera);
            glBindVertexArray(emptyVAO);
            glDrawArrays(GL_POINTS, 0, notOccludedSpheresCount);
            
            /// Cylinder drawing
            cylindersCullingShader.use();
            cylindersCullingShader.setInt("cylinderCount", cylinder_count);
            cylindersCullingShader.setUint("indexOffset", sphere_count);
            cylindersCullingShader.setVec2I("screenResolution", screenResolution);
            #if BENCHMARKING 

                cylindersCullingShader.setInt("benchmark", 1); 
            #else
                cylindersCullingShader.setInt("benchmark", 0);
            #endif
            setCameraUniforms(cylindersCullingShader, camera);
            setFrustumUniforms(cylindersCullingShader, frustum);
            glDispatchCompute(numGroupsXCylinders, numGroupsY, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

            #if BENCHMARKING 

                frustumEntitiesAtomicCounter.bind();
                glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &visibleCylindersCount);
                glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
                frustumEntitiesAtomicCounter.unbind(); 
                
                visibleEntitiesAtomicCounter.bind();
                glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &notOccludedCylindersCount);
                glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
                visibleEntitiesAtomicCounter.unbind();
            #else
            
                visibleEntitiesAtomicCounter.bind();
                glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &notOccludedCylindersCount);
                glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
                visibleCylindersCount = notOccludedCylindersCount;
                visibleEntitiesAtomicCounter.unbind();
            #endif

            cylindersShader.use();
            cylindersShader.setUint("indexOffset", sphere_count);
            cylindersShader.setVec2I("screenResolution", screenResolution);
            setCameraUniforms(cylindersShader, camera);
            glBindVertexArray(emptyVAO);
            glDrawArrays(GL_POINTS, 0, notOccludedCylindersCount);

            glBindTextureUnit(0, 0);

            canvasPixelIdTexture.bind();
            pixelCountShader.use();
            pixelCountShader.setVec2I("screenResolution", screenResolution);
            glDispatchCompute((SCR_WIDTH + workGroupSizeXPerPixel - 1) / workGroupSizeXPerPixel, 
                (SCR_HEIGHT + workGroupSizeYPerPixel - 1) / workGroupSizeYPerPixel, 
                1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            canvasPixelIdTexture.unbind();
            ////////////////
            glBindFramebuffer(GL_READ_FRAMEBUFFER, canvasFBO.getId());
            glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Bind Draw Framebuffer");
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
            glPopDebugGroup();
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

    ComputeShader spheresCullingShader("assets/shaders/standard_version/gpu_cull/spheresFrusCulling.compute", "Spheres Culling Shader");
    ShaderProgram spheresShader("assets/shaders/standard_version/gpu_cull/spheres.vert", 
                                "assets/shaders/standard_version/gpu_cull/spheres.frag",
                                "assets/shaders/standard_version/gpu_cull/spheres.geom",
                                "Spheres Program",
                                "Spheres Vertex Shader",
                                "Spheres Fragment Shader",
                                "Spheres Geometry Shader");
    spheresShader.linkProgram();
    
    ComputeShader cylindersCullingShader("assets/shaders/standard_version/gpu_cull/cylindersFrusCulling.compute", "Cylinders Culling Shader");
    ShaderProgram cylindersShader("assets/shaders/standard_version/gpu_cull/cylinders.vert", 
                                "assets/shaders/standard_version/gpu_cull/cylinders.frag",
                                "assets/shaders/standard_version/gpu_cull/cylinders.geom",
                                "Cylinders Program",
                                "Cylinders Vertex Shader",
                                "Cylinders Fragment Shader",
                                "Cylinders Geometry Shader");
    cylindersShader.linkProgram();

    std::vector<Sphere> spheres;
    std::vector<Cylinder> cylinders;

    getCompleteScene(spheres, sphere_count, cylinders, cylinder_count);

    sphere_count = spheres.size(); 
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

    std::cout << "Cylinders buffer " << std::endl;
    StorageBuffer cylinderBuffer(GL_SHADER_STORAGE_BUFFER, cylinder_count * sizeof(Cylinder), 4, 
        cylinders.data(), GL_STATIC_DRAW);

    std::cout << "Cylinder indexes buffer " << std::endl;
    std::vector<GLuint> visibleCylinders(cylinder_count, 0);
    StorageBuffer visibleCylindersBuffer(GL_SHADER_STORAGE_BUFFER, cylinder_count * sizeof(GLuint), 5, 
        visibleCylinders.data(), GL_STATIC_DRAW);

    spheres.clear();
    visibleSpheres.clear();
    cylinders.clear();
    visibleCylinders.clear();

    GLuint zero = 0;
    GLuint resetValue = 0;
    StorageBuffer visibleEntitiesAtomicCounter(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint), 3, 
    &zero, GL_STATIC_DRAW);
        

    GLuint emptyVAO;
    glCreateVertexArrays(1, &emptyVAO);

    Benchmark benchmark(camera_controller, chkPoints);

    std::string base_path = "media/csv/standard_version/gpu_cull/"; 
    base_path += ((currentScene == SceneType::LOADED_SCENE) ? ("loaded_scene_" + scene_file) : "packed_scene");
    std::string frame_times_path = base_path + "_frame_times.csv";
    std::string process_times_path = base_path + "_process_times.csv";
    Profiler profiler(window, 
        frame_times_path, 
        process_times_path, sphere_count, cylinder_count, 0, timerDuration);

    window.setupSceneInfoGui("Scene Info", scene_data);
    window.setupCameraGui("Camera Info", &camera);
    window.setupInputInfoGui("General Input Info");
    window.setupBenchmarkInfoGui("Benchmark", &benchmark);

    GLuint numGroupsXSpheres = (sphere_count + workGroupSizeXPerSphere - 1) / workGroupSizeXPerSphere;
    GLuint numGroupsXCylinders = (cylinder_count + workGroupSizeXPerCylinder - 1) / workGroupSizeXPerCylinder;
    GLuint numGroupsY = 1;

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
        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Bind Draw Framebuffer");
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glPopDebugGroup();
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
            profiler.updateProfiler(benchmark.getCheckpointID(), visibleSpheresCount, notOccludedSpheresCount, visibleCylindersCount, notOccludedCylindersCount);

            if (window.getInput().isKeyDown(Key::T)) profiler.startSavingNextFrames(benchmark.getCheckpointID());
            
            Frustum frustum(camera);

            /// Spheres drawing
            spheresCullingShader.use();
            spheresCullingShader.setInt("sphereCount", sphere_count);
            setFrustumUniforms(spheresCullingShader, frustum);
            dispatchComputeShaderWithLabel(numGroupsXSpheres, numGroupsY, "Sphere Culling Shader", GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
            
            visibleEntitiesAtomicCounter.bind();
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &visibleSpheresCount);
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
            notOccludedSpheresCount = visibleSpheresCount;
            visibleEntitiesAtomicCounter.unbind();

            spheresShader.use();
            spheresShader.setVec2I("screenResolution", screenResolution);
            setCameraUniforms(spheresShader, camera);
            glBindVertexArray(emptyVAO);
            drawArraysWithLabel(GL_POINTS, 0, visibleSpheresCount, "Draw Spheres");
            
            /// Cylinder drawing
            cylindersCullingShader.use();
            cylindersCullingShader.setInt("cylinderCount", cylinder_count);
            setFrustumUniforms(cylindersCullingShader, frustum);
            dispatchComputeShaderWithLabel(numGroupsXCylinders, numGroupsY, "Cylinder Culling Shader", GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
            
            visibleEntitiesAtomicCounter.bind();
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &visibleCylindersCount);
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
            notOccludedCylindersCount = visibleCylindersCount;
            visibleEntitiesAtomicCounter.unbind();

            cylindersShader.use();
            cylindersShader.setVec2I("screenResolution", screenResolution);
            setCameraUniforms(cylindersShader, camera);
            glBindVertexArray(emptyVAO);
            drawArraysWithLabel(GL_POINTS, 0, visibleCylindersCount, "Draw Cylinders");

            ////////////////
            glBindFramebuffer(GL_READ_FRAMEBUFFER, canvasFBO.getId());
            glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Bind Draw Framebuffer");
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
            glPopDebugGroup();
            isRunning = window.update();
        } 
    } catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return;
    }
    
}