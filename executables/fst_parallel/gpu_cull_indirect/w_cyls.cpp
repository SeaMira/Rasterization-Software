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

// ==============================================================================
// First Parallel Version - GPU Frustum Culling with Indirect Dispatch
// ==============================================================================
// Esta versión separa el frustum culling de la rasterización en shaders
// diferentes, utilizando Indirect Dispatch para ejecutar el shader de
// rasterización solo con el número necesario de work groups basado en
// el resultado del culling.
// 
// El cálculo de numGroupsX para el Indirect Dispatch se realiza directamente
// en los shaders de frustum culling usando atomicMax.
// ==============================================================================

// Settings
int SCR_WIDTH = 1024;
int SCR_HEIGHT = 1024;
int sphere_count = 552008;
int cylinder_count = 519767;

std::string title = "First Parallel Version: Indirect Dispatch - GPU Frustum Culling"; 

bool shown = true;

GLuint workGroupSizeXPerPixel = 16;  // Threads-per-group (X) for pixel processing
GLuint workGroupSizeYPerPixel = 16;  // Threads-per-group (Y) for pixel processing

GLuint workGroupSizeXPerSphere = 256;    // Threads-per-group for sphere processing
GLuint workGroupSizeXPerCylinder = 256;  // Threads-per-group for cylinder processing

int visibleSpheresCount = 0;
int visibleCylindersCount = 0;

double timerDuration = 48.0;

std::chrono::steady_clock::time_point startTime = std::chrono::high_resolution_clock::now();

void loadConfiguration()
{
    SceneSettings settings = SceneConfigLoader::loadDefault();
    SceneConfigLoader::applyToGlobals(settings);
    
    SCR_WIDTH = settings.screenWidth;
    SCR_HEIGHT = settings.screenHeight;
    sphere_count = settings.sphereCount;
    cylinder_count = settings.cylinderCount;
    workGroupSizeXPerPixel = settings.workGroupSizePerPixelX;
    workGroupSizeYPerPixel = settings.workGroupSizePerPixelY;
    workGroupSizeXPerSphere = settings.workGroupSizePerSphere;
    workGroupSizeXPerCylinder = settings.workGroupSizePerCylinder;
    timerDuration = settings.timerDuration;
}

int main(int argc, char* argv[]) 
{
    loadConfiguration();
    std::cout << "Starting Indirect Dispatch application..." << std::endl;
    startTime = std::chrono::high_resolution_clock::now();
    AppOpenGL window { title, SCR_WIDTH, SCR_HEIGHT, shown };

    Camera camera(SCR_WIDTH, SCR_HEIGHT);
    camera.SetPosition(.0f, .0f, .0f);

    std::unordered_map<std::string, int*> scene_data = {
        {"Screen width", &SCR_WIDTH},
        {"Screen height", &SCR_HEIGHT},
        {"Sphere count", &sphere_count},
        {"Visible Spheres", &visibleSpheresCount},
        {"Cylinder count", &cylinder_count},
        {"Visible Cylinders", &visibleCylindersCount}
    };

    CameraController camera_controller(window, camera);
    
    // ===========================================================================
    // Shaders
    // ===========================================================================
    
    // Shader de limpieza (resetea buffers y contadores)
    ComputeShader cleaningShader(
        "assets/shaders/fst_parallel_attempt/gpu_cull_indirect/set_to_black.compute", 
        "Cleaning Shader");
    
    // Shaders de frustum culling (generan listas de objetos visibles)
    ComputeShader sphereFrustumCullShader(
        "assets/shaders/fst_parallel_attempt/gpu_cull_indirect/sphere_frustum_cull.compute", 
        "Sphere Frustum Cull Shader");
    ComputeShader cylinderFrustumCullShader(
        "assets/shaders/fst_parallel_attempt/gpu_cull_indirect/cylinder_frustum_cull.compute", 
        "Cylinder Frustum Cull Shader");
    
    // Shaders de rasterización (procesan solo objetos visibles via Indirect Dispatch)
    ComputeShader sphereRasterizeShader(
        "assets/shaders/fst_parallel_attempt/gpu_cull_indirect/sphere_rasterize.compute", 
        "Sphere Rasterize Shader");
    ComputeShader cylinderRasterizeShader(
        "assets/shaders/fst_parallel_attempt/gpu_cull_indirect/cylinder_rasterize.compute", 
        "Cylinder Rasterize Shader");

    // ===========================================================================
    // Canvas and Framebuffer
    // ===========================================================================
    
    Canvas canvas(GL_TEXTURE_2D, GL_RGBA8, SCR_WIDTH, SCR_HEIGHT);
    canvas.setFBO(GL_COLOR_ATTACHMENT0);
    canvas.setupDepthData(SCR_WIDTH, SCR_HEIGHT);
    canvas.setupCleaningProgram(cleaningShader);
    
    // ===========================================================================
    // Scene Data
    // ===========================================================================
    
    std::vector<Sphere> spheres;
    std::vector<Cylinder> cylinders;
    getCompleteScene(spheres, sphere_count, cylinders, cylinder_count);
    const int actualSphereCount = static_cast<int>(spheres.size());
    const int actualCylinderCount = static_cast<int>(cylinders.size());

    std::vector<std::pair<glm::vec3, glm::vec3>> chkPoints = getCheckpoints(sphere_count, spheres, cylinders);
    
    visibleSpheresCount = actualSphereCount;
    visibleCylindersCount = actualCylinderCount;
    
    // Buffers de geometría
    StorageBuffer sphereBuffer(GL_SHADER_STORAGE_BUFFER, actualSphereCount * sizeof(Sphere), 6, 
        spheres.data(), GL_STATIC_DRAW);
    
    StorageBuffer cylinderBuffer(GL_SHADER_STORAGE_BUFFER, actualCylinderCount * sizeof(Cylinder), 7, 
        cylinders.data(), GL_STATIC_DRAW);

    spheres.clear();
    cylinders.clear();

    // ===========================================================================
    // Indirect Dispatch Buffers
    // ===========================================================================
    
    // Buffers para índices de objetos visibles
    StorageBuffer visibleSphereIndicesBuffer(GL_SHADER_STORAGE_BUFFER, 
        actualSphereCount * sizeof(GLuint), 10, nullptr, GL_DYNAMIC_COPY);
    
    StorageBuffer visibleCylinderIndicesBuffer(GL_SHADER_STORAGE_BUFFER, 
        actualCylinderCount * sizeof(GLuint), 13, nullptr, GL_DYNAMIC_COPY);
    
    // Contadores de objetos visibles
    GLuint zero = 0;
    StorageBuffer visibleSphereCountBuffer(GL_SHADER_STORAGE_BUFFER, 
        sizeof(GLuint), 11, &zero, GL_DYNAMIC_COPY);
    
    StorageBuffer visibleCylinderCountBuffer(GL_SHADER_STORAGE_BUFFER, 
        sizeof(GLuint), 14, &zero, GL_DYNAMIC_COPY);
    
    // Indirect Dispatch buffers (num_groups_x, num_groups_y, num_groups_z)
    // Estructura: { numGroupsX, numGroupsY, numGroupsZ } = 3 * sizeof(GLuint)
    GLuint initialDispatch[3] = { 1, 1, 1 };
    
    // Buffer para Indirect Dispatch de esferas (binding 12)
    GLuint sphereIndirectBuffer;
    glGenBuffers(1, &sphereIndirectBuffer);
    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, sphereIndirectBuffer);
    glBufferData(GL_DISPATCH_INDIRECT_BUFFER, 3 * sizeof(GLuint), initialDispatch, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 12, sphereIndirectBuffer);
    
    // Buffer para Indirect Dispatch de cilindros (binding 15)
    GLuint cylinderIndirectBuffer;
    glGenBuffers(1, &cylinderIndirectBuffer);
    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, cylinderIndirectBuffer);
    glBufferData(GL_DISPATCH_INDIRECT_BUFFER, 3 * sizeof(GLuint), initialDispatch, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 15, cylinderIndirectBuffer);
    
    // Contador para benchmarking
    GLuint resetValue = 0;
    StorageBuffer frustumAtomicCounter(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint), 8, 
        &zero, GL_STATIC_DRAW);

    // ===========================================================================
    // Benchmark and Profiler
    // ===========================================================================
    
    Benchmark benchmark(camera_controller, chkPoints);

    std::string base_path = "media/csv/fst_parallel_w_cyl/gpu_cull_indirect/"; 
    base_path += ((currentScene == SceneType::LOADED_SCENE) ? ("loaded_scene_" + scene_file) : "packed_scene");
    std::string frame_times_path = base_path + "_frame_times.csv";
    std::string process_times_path = base_path + "_process_times.csv";
    Profiler profiler(window, 
        frame_times_path, 
        process_times_path, actualSphereCount, actualCylinderCount, 0, timerDuration);

    window.setupSceneInfoGui("Scene Info", scene_data);
    window.setupCameraGui("Camera Info", &camera);
    window.setupInputInfoGui("General Input Info");
    window.setupBenchmarkInfoGui("Benchmark", &benchmark);
    
    // ===========================================================================
    // Work Group Calculations
    // ===========================================================================
    
    GLuint numGroupsXSpheres = (actualSphereCount + workGroupSizeXPerSphere - 1) / workGroupSizeXPerSphere;
    GLuint numGroupsXCylinders = (actualCylinderCount + workGroupSizeXPerCylinder - 1) / workGroupSizeXPerCylinder;
    GLuint numGroupsY = 1;

    canvas.bindTexture();
    canvas.bindFBO();

    glm::ivec2 screenResolution(SCR_WIDTH, SCR_HEIGHT);
    
    std::chrono::steady_clock::time_point endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = endTime - startTime;
    std::ofstream logFile("C:\\Users\\Sebatian\\Desktop\\XLIM\\Memoria_per_frame\\log.txt", std::ios::out);
    logFile << title << std::endl;
    logFile << "Elapsed time: " << elapsed_seconds.count() << " seconds" << std::endl;
    logFile.close();

    // ===========================================================================
    // Main Render Loop
    // ===========================================================================
    
    try
    {
        bool isRunning = true;
        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);

        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Bind Draw Framebuffer");
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glPopDebugGroup();
        
        while (isRunning)
        {
            camera_controller.cameraUpdate();
            benchmark.update();
            profiler.updateProfiler(benchmark.getCheckpointID(), 
                visibleSpheresCount, visibleSpheresCount,  // sin occlusion culling
                visibleCylindersCount, visibleCylindersCount);

            if (window.getInput().isKeyDown(Key::T)) 
                profiler.startSavingNextFrames(benchmark.getCheckpointID());

            // ===================================================================
            // Step 1: Clear buffers and reset counters
            // ===================================================================
            canvas.cleanCanvasBuffers(workGroupSizeXPerPixel, workGroupSizeYPerPixel, camera.getFar());

            Frustum frustum(camera);

            // ===================================================================
            // Step 2: Sphere Frustum Culling (includes atomicMax for indirect dispatch)
            // ===================================================================
            glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Sphere Frustum Culling");
            
            sphereFrustumCullShader.use();
            
            #if BENCHMARKING 
                frustumAtomicCounter.bind();
                glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &visibleCylindersCount);
                glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
                frustumAtomicCounter.unbind();
                sphereFrustumCullShader.setInt("benchmark", 1); 
            #else
                sphereFrustumCullShader.setInt("benchmark", 0);
            #endif
            
            sphereFrustumCullShader.setInt("sphereCount", sphere_count);
            sphereFrustumCullShader.setUint("workGroupSize", workGroupSizeXPerSphere);
            setFrustumUniforms(sphereFrustumCullShader, frustum);
            
            glDispatchCompute(numGroupsXSpheres, numGroupsY, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT);
            
            glPopDebugGroup();

            // ===================================================================
            // Step 3: Sphere Rasterization (Indirect Dispatch)
            // ===================================================================
            glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Sphere Rasterization (Indirect)");
            
            sphereRasterizeShader.use();
            sphereRasterizeShader.setVec2I("screenResolution", screenResolution);
            setCameraUniforms(sphereRasterizeShader, camera);
            
            // Indirect Dispatch basado en el número de esferas visibles
            glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, sphereIndirectBuffer);
            glDispatchComputeIndirect(0);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
            
            glPopDebugGroup();

            // ===================================================================
            // Step 4: Cylinder Frustum Culling (includes atomicMax for indirect dispatch)
            // ===================================================================
            glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Cylinder Frustum Culling");
            
            cylinderFrustumCullShader.use();
            
            #if BENCHMARKING 
                frustumAtomicCounter.bind();
                glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &visibleSpheresCount);
                glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
                frustumAtomicCounter.unbind();
                cylinderFrustumCullShader.setInt("benchmark", 1); 
            #else
                cylinderFrustumCullShader.setInt("benchmark", 0);
            #endif
            
            cylinderFrustumCullShader.setInt("cylinderCount", actualCylinderCount);
            cylinderFrustumCullShader.setUint("workGroupSize", workGroupSizeXPerCylinder);
            setFrustumUniforms(cylinderFrustumCullShader, frustum);
            
            glDispatchCompute(numGroupsXCylinders, numGroupsY, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT);
            
            glPopDebugGroup();

            // ===================================================================
            // Step 5: Cylinder Rasterization (Indirect Dispatch)
            // ===================================================================
            glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Cylinder Rasterization (Indirect)");
            
            cylinderRasterizeShader.use();
            cylinderRasterizeShader.setVec2I("screenResolution", screenResolution);
            setCameraUniforms(cylinderRasterizeShader, camera);
            
            // Indirect Dispatch basado en el número de cilindros visibles
            glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, cylinderIndirectBuffer);
            glDispatchComputeIndirect(0);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
            
            glPopDebugGroup();

            // ===================================================================
            // Step 6: Read back visible counts for UI (opcional, solo para debug)
            // ===================================================================
            #if BENCHMARKING
            glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
            
            visibleSphereCountBuffer.bind();
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &visibleSpheresCount);
            visibleSphereCountBuffer.unbind();
            
            visibleCylinderCountBuffer.bind();
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &visibleCylindersCount);
            visibleCylinderCountBuffer.unbind();
            #endif

            // ===================================================================
            // Step 7: Present frame
            // ===================================================================
            glBindFramebuffer(GL_READ_FRAMEBUFFER, canvas.getFramebuffer().getId());
            isRunning = window.update();

            if (window.getInput().isKeyDown(Key::F10)) 
            {
                std::string sshot_name = "media/img/fst_parallel_indirect/frame_" + 
                    std::to_string(benchmark.getCheckpointID()) + ".bmp";
                canvas.takeScreenshot(sshot_name);
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    // Cleanup
    glDeleteBuffers(1, &sphereIndirectBuffer);
    glDeleteBuffers(1, &cylinderIndirectBuffer);

    return 0;
}
