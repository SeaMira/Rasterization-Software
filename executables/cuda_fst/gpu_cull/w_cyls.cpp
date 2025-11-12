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
#include "vis/gl/cu/storage_buffer_cu.h"
#include "vis/gl/texture.h"
#include "vis/gl/cu/canvas_cu.h"

using uint = unsigned int;

// Settings
int SCR_WIDTH = 1024;
int SCR_HEIGHT = 1024;
int sphere_count = 4000000;
int cylinder_count = 4000000;

std::string title = "First Parallel CUDA Version: Spheres and Cylinders - GPU Frustum Culling"; 

bool shown = true;
bool withOcclusionCulling = false;

GLuint workGroupSizeXPerPixel = 16;  // Deifining threads-per-group (X)
GLuint workGroupSizeYPerPixel = 16;  // Deifining threads-per-group (Y)

GLuint workGroupSizeXPixelCount = 16;  // Deifining threads-per-group (X)
GLuint workGroupSizeYPixelCount = 16;  // Deifining threads-per-group (Y)

GLuint workGroupSizeXPerSphere = 256;  // Deifining threads-per-sphere (X)
GLuint workGroupSizeXPerCylinder = 256;  // Deifining threads-per-sphere (X)

int visibleSpheresCount = 0;
int notOccludedSpheresCount = 0;

int visibleCylindersCount = 0;
int notOccludedCylindersCount = 0;

// downsample settings
int downsampleLevel = 4;
int downsampleWorkGroupSizeX = (1 << downsampleLevel);
int downsampleWorkGroupSizeY = (1 << downsampleLevel);

std::chrono::steady_clock::time_point startTime = std::chrono::high_resolution_clock::now();

// void mainWithoutOcclusionCulling(Camera& camera, AppOpenGL& window);

#ifdef __cplusplus
extern "C" {
#endif
void cleaningScreen(
    cudaSurfaceObject_t texSurfaceObj, 
    unsigned int* depthBuffer,
    unsigned int* pixelOwnershipBuffer,
    int workGroupSizeXPerPixel,
    int workGroupSizeYPerPixel,
    float far, 
    int screenResolutionX,
    int screenResolutionY
    );

void downsamplingDepthTexture(
    unsigned int* depthBuffer, 
    cudaSurfaceObject_t downsampleSurface, 
    int screenResolutionX, 
    int screenResolutionY, 
    int downsampleWorkGroupSizeX, 
    int downsampleWorkGroupSizeY);

void sphereRaster(
    Sphere* d_spheres,
    int sphereCount,
    glm::mat4 view,
    glm::mat4 proj,
    glm::vec4 frustumTopFace,
    glm::vec4 frustumBottomFace,
    glm::vec4 frustumRightFace,
    glm::vec4 frustumLeftFace,
    glm::vec4 frustumFarFace,
    glm::vec4 frustumNearFace,
    glm::vec3 front,
    glm::vec3 up,
    glm::vec3 rightVec,
    glm::vec3 cameraPos,
    int screenW,
    int screenH,
    float fov,
    unsigned int* d_depthBuffer,
    unsigned int* d_pixelOwnershipBuffer,
    unsigned int* d_visibilityFrameBuffer,
    unsigned int visibilityFrameBufferIndexOffset,
    unsigned int* d_frustCullcounter,
    unsigned int* d_occCullcounter,
    int benchmark,
    cudaSurfaceObject_t outputImage,
    cudaTextureObject_t downsampleTex
);

void cylinderRaster(
    Cylinder* cylinders,
    int cylinderCount,
    glm::mat4 view,
    glm::mat4 proj,
    glm::vec4 frustumTopFace,
    glm::vec4 frustumBottomFace,
    glm::vec4 frustumRightFace,
    glm::vec4 frustumLeftFace,
    glm::vec4 frustumFarFace,
    glm::vec4 frustumNearFace,
    glm::vec3 front,
    glm::vec3 up,
    glm::vec3 rightVec,
    glm::vec3 cameraPos,
    int screenW,
    int screenH,
    float fov,
    unsigned int* d_depthBuffer,
    unsigned int* d_pixelOwnershipBuffer,
    unsigned int* d_visibilityFrameBuffer,
    unsigned int visibilityFrameBufferIndexOffset,
    unsigned int* d_frustCullcounter,
    unsigned int* d_occCullcounter,
    int benchmark,
    cudaSurfaceObject_t outputImage,
    cudaTextureObject_t downsampleTex
);

void pixelCount(
    unsigned int* pixelOwnershipBuffer,   // length = screenW*screenH
    unsigned int* d_visibilityFrameBuffer,
    int workGroupSizeX,
    int workGroupSizeY,
    int screenResolutionX,
    int screenResolutionY
    );
#ifdef __cplusplus
}
#endif


void mainWithOcclusionCulling(Camera& camera, AppOpenGL& window);

int main(int argc, char* argv[]) 
{
    startTime = std::chrono::high_resolution_clock::now();
    AppOpenGL window { title + (withOcclusionCulling ? " - With Occlusion Culling" : " - Without Occlusion Culling"), SCR_WIDTH, SCR_HEIGHT, shown };

    Camera camera(SCR_WIDTH, SCR_HEIGHT);
    camera.SetPosition(.0f, .0f, .0f);

    // withOcclusionCulling ? 
    //     mainWithOcclusionCulling(camera, window) :
    //     mainWithoutOcclusionCulling(camera, window);
    
    mainWithOcclusionCulling(camera, window);
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

    CanvasCUDA canvas(GL_TEXTURE_2D, GL_RGBA8, SCR_WIDTH, SCR_HEIGHT);
    canvas.setFBO(GL_COLOR_ATTACHMENT0);
    canvas.setupDepthDataCUDA(SCR_WIDTH, SCR_HEIGHT);
    canvas.setupDepthDownsampleCUDA(downsampleLevel, SCR_WIDTH, SCR_HEIGHT);
    unsigned int* depthBufferPtr = canvas.getDepthDataCUDA().getDepthBuffer();
    DepthDownsampleCUDA& canvasDepthDownsampleCUDA = canvas.getDepthDownsampleDataCUDA();
    
    std::cout << "Pixel Count Buffer" << std::endl;
    StorageBuffer pixelCountFramesBuffer(GL_SHADER_STORAGE_BUFFER, SCR_WIDTH*SCR_HEIGHT * sizeof(GLuint), 4, 
        nullptr, GL_DYNAMIC_COPY);
    StorageBufferCUDAWrapper pixelCountFramesBufferCUDAWrapper(pixelCountFramesBuffer);
    pixelCountFramesBufferCUDAWrapper.cudaMapResources();
    unsigned int* pixelOwnershipBufferPtr = pixelCountFramesBufferCUDAWrapper.getDevicePointer<unsigned int>();


    std::vector<Sphere> spheres = getScene(sphere_count);
    sphere_count = spheres.size(); 

    std::vector<Cylinder> cylinders = getCylinderScene(cylinder_count);
    cylinder_count = cylinders.size(); 

    std::vector<std::pair<glm::vec3, glm::vec3>> chkPoints = getCheckpoints(sphere_count, spheres, cylinders);
    
    visibleSpheresCount = sphere_count;
    
    visibleCylindersCount = cylinder_count;
    GLint maxSSBOsize = 0;
    glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &maxSSBOsize);
    std::cout << "Max SSBO size: " << maxSSBOsize << std::endl;
    
    std::cout << "Spheres Buffer" << std::endl;
    StorageBuffer sphereBuffer(GL_SHADER_STORAGE_BUFFER, sphere_count * sizeof(Sphere), 6, 
        spheres.data(), GL_STATIC_DRAW);
    StorageBufferCUDAWrapper sphereBufferCUDAWrapper(sphereBuffer);
    sphereBufferCUDAWrapper.cudaMapResources();
    Sphere * sphereBufferPtr = sphereBufferCUDAWrapper.getDevicePointer<Sphere>();

    std::cout << "Cylinders Buffer" << std::endl;
    StorageBuffer cylinderBuffer(GL_SHADER_STORAGE_BUFFER, cylinder_count * sizeof(Cylinder), 7, 
        cylinders.data(), GL_STATIC_DRAW);
    StorageBufferCUDAWrapper cylinderBufferCUDAWrapper(cylinderBuffer);
    cylinderBufferCUDAWrapper.cudaMapResources();
    Cylinder * cylinderBufferPtr = cylinderBufferCUDAWrapper.getDevicePointer<Cylinder>();

    // all entities visibility info ssbo
    std::cout << "Visibility Frames Buffer" << std::endl;
    int totalEntities = sphere_count + cylinder_count;
    std::vector<GLuint> visibilityFrames(2 * totalEntities, 10);
    StorageBuffer visibilityFramesBuffer(GL_SHADER_STORAGE_BUFFER, 2 * totalEntities * sizeof(GLuint), 5, 
    visibilityFrames.data(), GL_DYNAMIC_COPY);
    StorageBufferCUDAWrapper visibilityFramesBufferCUDAWrapper(visibilityFramesBuffer);
    visibilityFramesBufferCUDAWrapper.cudaMapResources();
    unsigned int* visibilityFrameBufferPtr = visibilityFramesBufferCUDAWrapper.getDevicePointer<unsigned int>();
    
    GLuint zero = 0;
    GLuint resetValue = 0;
    std::cout << "Frustum Atomic Counter" << std::endl;
    StorageBuffer frustumAtomicCounter(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint), 8, 
    &zero, GL_STATIC_DRAW);
    StorageBufferCUDAWrapper frustumAtomicCounterCUDAWrapper(frustumAtomicCounter);
    frustumAtomicCounterCUDAWrapper.cudaMapResources();
    unsigned int* frustumCounter = frustumAtomicCounterCUDAWrapper.getDevicePointer<unsigned int>();
    
    std::cout << "Occlusion Atomic Counter" << std::endl;
    StorageBuffer occlusionAtomicCounter(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint), 9, 
        &zero, GL_STATIC_DRAW);
    StorageBufferCUDAWrapper occlusionAtomicCounterCUDAWrapper(occlusionAtomicCounter);
    occlusionAtomicCounterCUDAWrapper.cudaMapResources();
    unsigned int* occlusionCounter = occlusionAtomicCounterCUDAWrapper.getDevicePointer<unsigned int>();

    Benchmark benchmark(camera_controller, chkPoints);

    std::string base_path = "media/csv/fst_parallel_w_cyl/gpu_cull/occ_"; 
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

            // canvas.downsampleCanvasDepth(downsampleWorkGroupSizeX, downsampleWorkGroupSizeY);
            
            // canvas.cleanCanvasBuffers(workGroupSizeXPerPixel, workGroupSizeYPerPixel, camera.getFar());

            Frustum frustum(camera);

            // spheresShader.use();
            // #if BENCHMARKING 
            //     frustumAtomicCounter.bind();
            //     glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &visibleCylindersCount);
            //     glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
            //     frustumAtomicCounter.unbind();

            //     occlusionAtomicCounter.bind();
            //     glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &notOccludedCylindersCount);
            //     glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
            //     occlusionAtomicCounter.unbind();

            //     spheresShader.setInt("benchmark", 1); 
            // #else
            //     spheresShader.setInt("benchmark", 0);
            // #endif
            // // spheresShader.setInt("sphereCount", sphere_count);
            // // spheresShader.setUint("visibilityFrameBufferIndexOffset", 0);
            // // spheresShader.setVec2I("screenResolution", screenResolution);
            // setFrustumUniforms(spheresShader, frustum);
            // setCameraUniforms(spheresShader, camera);
            // glDispatchCompute(numGroupsXSpheres, numGroupsY, 1);
            // glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
            
            // cylinderShader.use();
            // #if BENCHMARKING 
            //     frustumAtomicCounter.bind();
            //     glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &visibleSpheresCount);
            //     glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
            //     frustumAtomicCounter.unbind();

            //     occlusionAtomicCounter.bind();
            //     glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &notOccludedSpheresCount);
            //     glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
            //     occlusionAtomicCounter.unbind();
                
            //     cylinderShader.setInt("benchmark", 1); 
            // #else
            //     cylinderShader.setInt("benchmark", 0);
            // #endif
            // cylinderShader.setInt("cylinderCount", cylinder_count);
            // cylinderShader.setUint("visibilityFrameBufferIndexOffset", sphere_count);
            // cylinderShader.setVec2I("screenResolution", screenResolution);
            // setFrustumUniforms(cylinderShader, frustum);
            // setCameraUniforms(cylinderShader, camera);
            // glDispatchCompute(numGroupsXCylinders, numGroupsY, 1);
            // glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

            // pixelCountComputeShader.use();
            // pixelCountComputeShader.setVec2I("screenResolution", screenResolution);
            // glDispatchCompute((SCR_WIDTH + workGroupSizeXPerPixel - 1) / workGroupSizeXPerPixel, 
            //     (SCR_HEIGHT + workGroupSizeYPerPixel - 1) / workGroupSizeYPerPixel, 
            //     1);
            // glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            cudaMemset(frustumCounter, 0, sizeof(unsigned int));
            cudaMemset(occlusionCounter, 0, sizeof(unsigned int));

            downsamplingDepthTexture(
                depthBufferPtr,
                canvasDepthDownsampleCUDA.getSurface(),
                SCR_WIDTH,
                SCR_HEIGHT,
                downsampleWorkGroupSizeX,
                downsampleWorkGroupSizeY
            );


            TextureCUDAWrapper& canvasTextureCUDAWrapper = canvas.getCUDATextureWrapper();
            canvasTextureCUDAWrapper.cudaMapResources();
            canvasTextureCUDAWrapper.cudaCreateSurfaceObj();
            
            cleaningScreen(
                canvasTextureCUDAWrapper.getSurfaceObject(), 
                depthBufferPtr,
                pixelOwnershipBufferPtr,
                workGroupSizeXPerPixel,
                workGroupSizeYPerPixel,
                camera.getFar(),
                SCR_WIDTH,
                SCR_HEIGHT
            );


            sphereRaster(
                sphereBufferPtr,
                sphere_count,
                camera.getView(),
                camera.getProjection(),
                glm::vec4(frustum.topFace.normal.x, frustum.topFace.normal.y, frustum.topFace.normal.z, frustum.topFace.distance),
                glm::vec4(frustum.bottomFace.normal.x, frustum.bottomFace.normal.y, frustum.bottomFace.normal.z, frustum.bottomFace.distance),
                glm::vec4(frustum.rightFace.normal.x, frustum.rightFace.normal.y, frustum.rightFace.normal.z, frustum.rightFace.distance),
                glm::vec4(frustum.leftFace.normal.x, frustum.leftFace.normal.y, frustum.leftFace.normal.z, frustum.leftFace.distance),
                glm::vec4(frustum.farFace.normal.x, frustum.farFace.normal.y, frustum.farFace.normal.z, frustum.farFace.distance),
                glm::vec4(frustum.nearFace.normal.x, frustum.nearFace.normal.y, frustum.nearFace.normal.z, frustum.nearFace.distance),
                camera.getFront(),
                camera.getUp(),
                camera.getRight(),
                camera.getPosition(),
                SCR_WIDTH,
                SCR_HEIGHT,
                camera.getFov(),
                depthBufferPtr,
                pixelOwnershipBufferPtr,
                visibilityFrameBufferPtr,
                0,
                frustumCounter,
                occlusionCounter,
                1,
                canvasTextureCUDAWrapper.getSurfaceObject(),
                canvasDepthDownsampleCUDA.getTexture()
            );

            unsigned int h_frustCount = 0;
            cudaMemcpy(&h_frustCount, frustumCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            visibleSpheresCount = h_frustCount;
            
            unsigned int h_occCount = 0;
            cudaMemcpy(&h_occCount, occlusionCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            notOccludedSpheresCount = h_occCount;


            cudaMemset(frustumCounter, 0, sizeof(unsigned int));
            cudaMemset(occlusionCounter, 0, sizeof(unsigned int));

            cylinderRaster(
                cylinderBufferPtr,
                cylinder_count,
                camera.getView(),
                camera.getProjection(),
                glm::vec4(frustum.topFace.normal.x, frustum.topFace.normal.y, frustum.topFace.normal.z, frustum.topFace.distance),
                glm::vec4(frustum.bottomFace.normal.x, frustum.bottomFace.normal.y, frustum.bottomFace.normal.z, frustum.bottomFace.distance),
                glm::vec4(frustum.rightFace.normal.x, frustum.rightFace.normal.y, frustum.rightFace.normal.z, frustum.rightFace.distance),
                glm::vec4(frustum.leftFace.normal.x, frustum.leftFace.normal.y, frustum.leftFace.normal.z, frustum.leftFace.distance),
                glm::vec4(frustum.farFace.normal.x, frustum.farFace.normal.y, frustum.farFace.normal.z, frustum.farFace.distance),
                glm::vec4(frustum.nearFace.normal.x, frustum.nearFace.normal.y, frustum.nearFace.normal.z, frustum.nearFace.distance),
                camera.getFront(),
                camera.getUp(),
                camera.getRight(),
                camera.getPosition(),
                SCR_WIDTH,
                SCR_HEIGHT,
                camera.getFov(),
                depthBufferPtr,
                pixelOwnershipBufferPtr,
                visibilityFrameBufferPtr,
                sphere_count,
                frustumCounter,
                occlusionCounter,
                1,
                canvasTextureCUDAWrapper.getSurfaceObject(),
                canvasDepthDownsampleCUDA.getTexture()
            );

            cudaMemcpy(&h_frustCount, frustumCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            visibleCylindersCount = h_frustCount;
            
            cudaMemcpy(&h_occCount, occlusionCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            notOccludedCylindersCount = h_occCount;

            pixelCount(
                pixelOwnershipBufferPtr,
                visibilityFrameBufferPtr,
                workGroupSizeXPixelCount,
                workGroupSizeYPixelCount,
                SCR_WIDTH,
                SCR_HEIGHT
            );

            
            canvasTextureCUDAWrapper.cudaDestroySurfaceObj();
            canvasTextureCUDAWrapper.cudaUnmapResources();

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
    pixelCountFramesBufferCUDAWrapper.cudaUnmapResources();
    sphereBufferCUDAWrapper.cudaUnmapResources();
    cylinderBufferCUDAWrapper.cudaUnmapResources();
    visibilityFramesBufferCUDAWrapper.cudaUnmapResources();
    frustumAtomicCounterCUDAWrapper.cudaUnmapResources();
    occlusionAtomicCounterCUDAWrapper.cudaUnmapResources();
}

// void mainWithoutOcclusionCulling(Camera& camera, AppOpenGL& window)
// {
//     std::unordered_map<std::string, int*> scene_data = {
//         {"Screen width", &SCR_WIDTH},
//         {"Screen height", &SCR_HEIGHT},
//         {"Sphere count", &sphere_count},
//         {"Spheres On Frustum", &visibleSpheresCount},
//         {"Drawn spheres", &notOccludedSpheresCount},
//         {"Cylinder count", &cylinder_count},
//         {"Cylinders On Frustum", &visibleCylindersCount},
//         {"Drawn cylinders", &notOccludedCylindersCount}
//     };

//     CameraController camera_controller(window, camera);
    
//     ComputeShader spheresShader("assets/shaders/fst_parallel_attempt/gpu_cull/sphere_culling_no_occ.compute", "Spheres Shader");
//     ComputeShader cylinderShader("assets/shaders/fst_parallel_attempt/gpu_cull/cylinder_culling_no_occ.compute", "Cylinders Shader");
//     ComputeShader cleaningComputeShader("assets/shaders/fst_parallel_attempt/gpu_cull/set_to_black_no_occ.compute", "Cleaning Shader");

//     Canvas canvas(GL_TEXTURE_2D, GL_RGBA8, SCR_WIDTH, SCR_HEIGHT);
//     canvas.setFBO(GL_COLOR_ATTACHMENT0);
//     canvas.setupDepthData(SCR_WIDTH, SCR_HEIGHT);
//     canvas.setupCleaningProgram(cleaningComputeShader);
    
//     std::vector<Sphere> spheres = getScene(sphere_count);
//     sphere_count = spheres.size(); 

//     std::vector<Cylinder> cylinders = getCylinderScene(cylinder_count);
//     cylinder_count = cylinders.size(); 

//     std::vector<std::pair<glm::vec3, glm::vec3>> chkPoints = getCheckpoints(sphere_count, spheres, cylinders);
    
//     visibleSpheresCount = sphere_count;
    
//     visibleCylindersCount = cylinder_count;
    
//     StorageBuffer sphereBuffer(GL_SHADER_STORAGE_BUFFER, sphere_count * sizeof(Sphere), 6, 
//         spheres.data(), GL_STATIC_DRAW);
    
//     StorageBuffer cylinderBuffer(GL_SHADER_STORAGE_BUFFER, cylinder_count * sizeof(Cylinder), 7, 
//         cylinders.data(), GL_STATIC_DRAW);
    

//     GLuint zero = 0;
//     GLuint resetValue = 0;
//     StorageBuffer frustumAtomicCounter(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint), 8, 
//         &zero, GL_STATIC_DRAW);

//     Benchmark benchmark(camera_controller, chkPoints);

//     std::string base_path = "media/csv/fst_parallel_w_cyl/gpu_cull/"; 
//     base_path += ((currentScene == SceneType::LOADED_SCENE) ? ("loaded_scene_" + scene_file) : "packed_scene");
//     std::string frame_times_path = base_path + "_frame_times.csv";
//     std::string process_times_path = base_path + "_process_times.csv";
//     Profiler profiler(window, 
//         frame_times_path, 
//         process_times_path, sphere_count, cylinder_count, 0, 48.0f);

//     window.setupSceneInfoGui("Scene Info", scene_data);
//     window.setupCameraGui("Camera Info", &camera);
//     window.setupInputInfoGui("General Input Info");
//     window.setupBenchmarkInfoGui("Benchmark", &benchmark);
    
//     // Calculating number of work groups (based on the number of threads and spheres)
//     GLuint numGroupsXSpheres = (sphere_count + workGroupSizeXPerSphere - 1) / workGroupSizeXPerSphere;
//     GLuint numGroupsXCylinders = (cylinder_count + workGroupSizeXPerCylinder - 1) / workGroupSizeXPerCylinder;
//     GLuint numGroupsY = 1;

//     canvas.bindTexture();
//     canvas.bindFBO();

//     glm::ivec2 screenResolution(SCR_WIDTH, SCR_HEIGHT);
//     std::chrono::steady_clock::time_point endTime = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> elapsed_seconds = endTime - startTime;
//     std::ofstream logFile("C:\\Users\\Sebatian\\Desktop\\XLIM\\Memoria_per_frame\\log.txt", std::ios::out);
//     logFile << title << std::endl;
//     logFile << "Elapsed time: " << elapsed_seconds.count() << " seconds" << std::endl;
//     logFile << "With occlusion culling: " << withOcclusionCulling << std::endl;
//     logFile.close();

    
//     try
//     {
//         bool isRunning = true;
//         while ( isRunning )
//         {
//             camera_controller.cameraUpdate();
//             benchmark.update();
//             profiler.updateProfiler(benchmark.getCheckpointID(), visibleSpheresCount, notOccludedSpheresCount, visibleCylindersCount, notOccludedCylindersCount);

//             if (window.getInput().isKeyDown(Key::T)) profiler.startSavingNextFrames(benchmark.getCheckpointID());
            
//             canvas.cleanCanvasBuffers(workGroupSizeXPerPixel, workGroupSizeYPerPixel, camera.getFar());

//             Frustum frustum(camera);

//             spheresShader.use();
//             #if BENCHMARKING 
//                 frustumAtomicCounter.bind();
//                 glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &visibleCylindersCount);
//                 glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
//                 frustumAtomicCounter.unbind();
//                 notOccludedCylindersCount = visibleCylindersCount;
//                 spheresShader.setInt("benchmark", 1); 
//             #else
//                 spheresShader.setInt("benchmark", 0);
//             #endif
//             spheresShader.setInt("sphereCount", sphere_count);
//             spheresShader.setVec2I("screenResolution", screenResolution);
//             setFrustumUniforms(spheresShader, frustum);
//             setCameraUniforms(spheresShader, camera);
//             glDispatchCompute(numGroupsXSpheres, numGroupsY, 1);
//             glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
            
//             cylinderShader.use();
//             #if BENCHMARKING 
//                 frustumAtomicCounter.bind();
//                 glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &visibleSpheresCount);
//                 glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
//                 frustumAtomicCounter.unbind();
//                 notOccludedSpheresCount = visibleSpheresCount;
//                 cylinderShader.setInt("benchmark", 1); 
//             #else
//                 cylinderShader.setInt("benchmark", 0);
//             #endif
//             cylinderShader.setInt("cylinderCount", cylinder_count);
//             cylinderShader.setVec2I("screenResolution", screenResolution);
//             setFrustumUniforms(cylinderShader, frustum);
//             setCameraUniforms(cylinderShader, camera);
//             glDispatchCompute(numGroupsXCylinders, numGroupsY, 1);
//             glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
            
//             glBindFramebuffer(GL_READ_FRAMEBUFFER, canvas.getFramebuffer().getId());
//             isRunning = window.update();

//             ////// checking drawn spheres ///////
            
//             // sphereBuffer.bind();  // Primero aseguramos que el buffer está vinculado

//             // // mapping buffer on reading mode
//             // void* mappedData = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 
//             //                                     0, 
//             //                                     spheres.size() * sizeof(SphereContainer), 
//             //                                     GL_MAP_READ_BIT);

//             // // verifying correct mapping
//             // if (mappedData != nullptr) {
//             //     // Accessing mapped buffer
//             //     SphereContainer* spheresData = reinterpret_cast<SphereContainer*>(mappedData);
//             //     notOccludedSpheresCount = 0;
//             //     // looping on elements checking if drawn or not
//             //     for (size_t i = 0; i < visibleSpheresCount; ++i)
//             //         if (spheresData[i].wasDrawn[0] == 1) 
//             //             notOccludedSpheresCount++;
                
//             //     // unmapping buffer once finished
//             //     glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
//             // } else {
//             //     std::cerr << "Failed to map the buffer!" << std::endl;
//             // }
            
//             ////// END:: checking drawn spheres ///////
            
//             if (window.getInput().isKeyDown(Key::F10)) 
//             {
//                 std::string sshot_name  = "media/img/fst_parallel/frame_" + std::to_string(benchmark.getCheckpointID()) + ".bmp";
//                 canvas.takeScreenshot(sshot_name);
//             }
//         }
//     }
//     catch (const std::exception& e)
//     {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return ;
//     }
// }