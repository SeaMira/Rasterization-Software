#include <iostream>
#include <fstream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glad/glad.h>
#include <SDL3/SDL.h>
#include <vector>
#include <nvtx3/nvToolsExt.h>

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
#include "vis/gl/cu/storage_buffer_cu.h"
#include "vis/gl/texture.h"
#include "vis/gl/cu/canvas_cu.h"


using uint = unsigned int;

// Settings
int SCR_WIDTH = 1024;
int SCR_HEIGHT = 1024;
int sphere_count = 126;
int cylinder_count = 142;

std::string title = "First Parallel CUDA Version: Spheres and Cylinders - GPU Frustum Culling"; 

bool shown = true;
bool withOcclusionCulling = false;

GLuint workGroupSizeXPerPixel = 16;  // Deifining threads-per-group (X)
GLuint workGroupSizeYPerPixel = 16;  // Deifining threads-per-group (Y)

GLuint workGroupSizeXPixelCount = 32;  // Deifining threads-per-group (X)
GLuint workGroupSizeYPixelCount = 32;  // Deifining threads-per-group (Y)

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

double timerDuration = 48.0;

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
    float c_far, 
    int c_screenResolutionX,
    int c_screenResolutionY,
    cudaStream_t& stream
    );


void cleaningScreenNoOcc(
    cudaSurfaceObject_t texSurfaceObj, 
    unsigned int* depthBuffer,
    int workGroupSizeXPerPixel,
    int workGroupSizeYPerPixel,
    float c_far, 
    int c_screenResolutionX,
    int c_screenResolutionY,
    cudaStream_t& stream
    );

void downsamplingDepthTexture(
    unsigned int* depthBuffer, 
    cudaSurfaceObject_t downsampleSurface, 
    int c_screenResolutionX, 
    int c_screenResolutionY, 
    int downsampleWorkGroupSizeX, 
    int downsampleWorkGroupSizeY,
    cudaStream_t& stream
);

void sphereRaster(
    glm::vec4* d_spheres,
    int c_sphereCount,
    glm::mat4 c_view,
    glm::mat4 c_proj,
    glm::vec4 c_frustumTopFace,
    glm::vec4 c_frustumBottomFace,
    glm::vec4 c_frustumRightFace,
    glm::vec4 c_frustumLeftFace,
    glm::vec4 c_frustumFarFace,
    glm::vec4 c_frustumNearFace,
    glm::vec3 c_front,
    glm::vec3 c_up,
    glm::vec3 c_right,
    glm::vec3 c_cameraPos,
    int c_screenW,
    int c_screenH,
    float c_fov,
    unsigned int* d_depthBuffer,
    unsigned int* d_pixelOwnershipBuffer,
    unsigned int* d_visibilityFrameBuffer,
    unsigned int c_visibilityFrameBufferIndexOffset,
    unsigned int* d_frustCullcounter,
    unsigned int* d_occCullcounter,
    int c_benchmark,
    cudaSurfaceObject_t outputImage,
    cudaTextureObject_t downsampleTex,
    cudaStream_t& stream,
    int workGroupSizeX
);

void sphereRasterNoOcc(
    glm::vec4* d_spheres,
    int c_sphereCount,
    glm::mat4 c_view,
    glm::mat4 c_proj,
    glm::vec4 c_frustumTopFace,
    glm::vec4 c_frustumBottomFace,
    glm::vec4 c_frustumRightFace,
    glm::vec4 c_frustumLeftFace,
    glm::vec4 c_frustumFarFace,
    glm::vec4 c_frustumNearFace,
    glm::vec3 c_front,
    glm::vec3 c_up,
    glm::vec3 c_right,
    glm::vec3 c_cameraPos,
    int c_screenW,
    int c_screenH,
    float c_fov,
    unsigned int* d_depthBuffer,
    unsigned int* d_frustCullcounter,
    int c_benchmark,
    cudaSurfaceObject_t outputImage,
    cudaStream_t& stream,
    int workGroupSizeX
);

void cylinderRaster(
    Cylinder* cylinders,
    int c_cylinderCount,
    glm::mat4 c_view,
    glm::mat4 c_proj,
    glm::vec4 c_frustumTopFace,
    glm::vec4 c_frustumBottomFace,
    glm::vec4 c_frustumRightFace,
    glm::vec4 c_frustumLeftFace,
    glm::vec4 c_frustumFarFace,
    glm::vec4 c_frustumNearFace,
    glm::vec3 c_front,
    glm::vec3 c_up,
    glm::vec3 c_right,
    glm::vec3 c_cameraPos,
    int c_screenW,
    int c_screenH,
    float c_fov,
    unsigned int* d_depthBuffer,
    unsigned int* d_pixelOwnershipBuffer,
    unsigned int* d_visibilityFrameBuffer,
    unsigned int c_visibilityFrameBufferIndexOffset,
    unsigned int* d_frustCullcounter,
    unsigned int* d_occCullcounter,
    int c_benchmark,
    cudaSurfaceObject_t outputImage,
    cudaTextureObject_t downsampleTex,
    cudaStream_t& stream,
    int workGroupSizeX
);

void cylinderRasterNoOcc(
    Cylinder* cylinders,
    int c_cylinderCount,
    glm::mat4 c_view,
    glm::mat4 c_proj,
    glm::vec4 c_frustumTopFace,
    glm::vec4 c_frustumBottomFace,
    glm::vec4 c_frustumRightFace,
    glm::vec4 c_frustumLeftFace,
    glm::vec4 c_frustumFarFace,
    glm::vec4 c_frustumNearFace,
    glm::vec3 c_front,
    glm::vec3 c_up,
    glm::vec3 c_right,
    glm::vec3 c_cameraPos,
    int c_screenW,
    int c_screenH,
    float c_fov,
    unsigned int* d_depthBuffer,
    unsigned int* d_frustCullcounter,
    int c_benchmark,
    cudaSurfaceObject_t outputImage,
    cudaStream_t& stream,
    int workGroupSizeX
);

void pixelCount(
    unsigned int* pixelOwnershipBuffer,   // length = screenW*screenH
    unsigned int* d_visibilityFrameBuffer,
    int workGroupSizeX,
    int workGroupSizeY,
    int c_screenResolutionX,
    int c_screenResolutionY,
    cudaStream_t& stream
    );
#ifdef __cplusplus
}
#endif

void loadConfiguration() {
    SceneConfigLoader configLoader;
    if (configLoader.loadConfig("assets/config/scene_config.json")) {
        SCR_WIDTH = configLoader.getScreenWidth();
        SCR_HEIGHT = configLoader.getScreenHeight();
        timerDuration = configLoader.getTimerDuration();
        downsampleLevel = configLoader.getDownsampleLevel();
        downsampleWorkGroupSizeX = (1 << downsampleLevel);
        downsampleWorkGroupSizeY = (1 << downsampleLevel);
    }
}

void mainWithOcclusionCulling(Camera& camera, AppOpenGL& window);
void mainWithoutOcclusionCulling(Camera& camera, AppOpenGL& window);

int main(int argc, char* argv[]) 
{
    loadConfiguration();
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

    CanvasCUDA canvas[2] = {CanvasCUDA(GL_TEXTURE_2D, GL_RGBA8, SCR_WIDTH, SCR_HEIGHT),
                            CanvasCUDA(GL_TEXTURE_2D, GL_RGBA8, SCR_WIDTH, SCR_HEIGHT)};
    canvas[0].setFBO(GL_COLOR_ATTACHMENT0);
    canvas[0].setupDepthDataCUDA(SCR_WIDTH, SCR_HEIGHT);
    canvas[0].setupDepthDownsampleCUDA(downsampleLevel, SCR_WIDTH, SCR_HEIGHT);
    unsigned int* depthBufferPtr = canvas[0].getDepthDataCUDA().getDepthBuffer();
    DepthDownsampleCUDA& canvasDepthDownsampleCUDA = canvas[0].getDepthDownsampleDataCUDA();

    canvas[1].setFBO(GL_COLOR_ATTACHMENT0);
    
        auto cerr = cudaGetLastError();
    if (cerr != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(cerr));
    GLenum glerr = glGetError();
    if (glerr) std::cout << "GL error: 0x" << std::hex << glerr << std::dec << std::endl;

    std::cout << "Pixel Count Buffer" << std::endl;
    StorageBuffer pixelCountFramesBuffer(GL_SHADER_STORAGE_BUFFER, SCR_WIDTH*SCR_HEIGHT * sizeof(GLuint), 4, 
        nullptr, GL_DYNAMIC_COPY);
    StorageBufferCUDAWrapper pixelCountFramesBufferCUDAWrapper(pixelCountFramesBuffer);
    pixelCountFramesBufferCUDAWrapper.cudaMapResources();
    unsigned int* pixelOwnershipBufferPtr = pixelCountFramesBufferCUDAWrapper.getDevicePointer<unsigned int>();


    std::vector<Sphere> spheres;
    std::vector<Cylinder> cylinders;
    getCompleteScene(spheres, sphere_count, cylinders, cylinder_count);

    sphere_count = spheres.size(); 
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
    std::cout << "Frustum Spheres Atomic Counter" << std::endl;
    StorageBuffer frustumSpheresAtomicCounter(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint), 8, 
    &zero, GL_STATIC_DRAW);
    StorageBufferCUDAWrapper frustumSpheresAtomicCounterCUDAWrapper(frustumSpheresAtomicCounter);
    frustumSpheresAtomicCounterCUDAWrapper.cudaMapResources();
    unsigned int* frustumSpheresCounter = frustumSpheresAtomicCounterCUDAWrapper.getDevicePointer<unsigned int>();
    
    std::cout << "Occlusion Spheres Atomic Counter" << std::endl;
    StorageBuffer occlusionSpheresAtomicCounter(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint), 9, 
        &zero, GL_STATIC_DRAW);
    StorageBufferCUDAWrapper occlusionSpheresAtomicCounterCUDAWrapper(occlusionSpheresAtomicCounter);
    occlusionSpheresAtomicCounterCUDAWrapper.cudaMapResources();
    unsigned int* occlusionSpheresCounter = occlusionSpheresAtomicCounterCUDAWrapper.getDevicePointer<unsigned int>();
    
    std::cout << "Frustum Cylinders Atomic Counter" << std::endl;
    StorageBuffer frustumCylindersAtomicCounter(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint), 8, 
    &zero, GL_STATIC_DRAW);
    StorageBufferCUDAWrapper frustumCylindersAtomicCounterCUDAWrapper(frustumCylindersAtomicCounter);
    frustumCylindersAtomicCounterCUDAWrapper.cudaMapResources();
    unsigned int* frustumCylindersCounter = frustumCylindersAtomicCounterCUDAWrapper.getDevicePointer<unsigned int>();
    
    std::cout << "Occlusion Cylinders Atomic Counter" << std::endl;
    StorageBuffer occlusionCylindersAtomicCounter(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint), 9, 
        &zero, GL_STATIC_DRAW);
    StorageBufferCUDAWrapper occlusionCylindersAtomicCounterCUDAWrapper(occlusionCylindersAtomicCounter);
    occlusionCylindersAtomicCounterCUDAWrapper.cudaMapResources();
    unsigned int* occlusionCylindersCounter = occlusionCylindersAtomicCounterCUDAWrapper.getDevicePointer<unsigned int>();
    
    // pinned memory for readback
    unsigned int* visibleSpheresCountPinned;
    unsigned int* notOccludedSpheresCountPinned;
    unsigned int* visibleCylindersCountPinned;
    unsigned int* notOccludedCylindersCountPinned;

    cudaHostAlloc(&visibleSpheresCountPinned, sizeof(unsigned int), cudaHostAllocDefault);
    cudaHostAlloc(&notOccludedSpheresCountPinned, sizeof(unsigned int), cudaHostAllocDefault);
    cudaHostAlloc(&visibleCylindersCountPinned, sizeof(unsigned int), cudaHostAllocDefault);
    cudaHostAlloc(&notOccludedCylindersCountPinned, sizeof(unsigned int), cudaHostAllocDefault);

    Benchmark benchmark(camera_controller, chkPoints);

    std::string base_path = "media/csv/fst_parallel_w_cyl/gpu_cull/occ_"; 
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
    
    std::chrono::steady_clock::time_point endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = endTime - startTime;
    std::ofstream logFile("C:\\Users\\Sebatian\\Desktop\\XLIM\\Memoria_per_frame\\log.txt", std::ios::out);

    int count = 0;
    cudaGetDeviceCount(&count);

    for (int i = 0; i < count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "Device " << i << ": " << prop.name << std::endl;
        
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Memoria global: " << prop.totalGlobalMem / (1024.0 * 1024 * 1024) << " GB" << std::endl;
    }
    cudaSetDevice(0);
    
    try
    {
        bool fbo1o2 = false;
        bool isRunning = true;

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        cudaEvent_t evtReady[2];
        cudaEventCreateWithFlags(&evtReady[0], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&evtReady[1], cudaEventDisableTiming);

        TextureCUDAWrapper* canvasCUDA[2] = {
            &canvas[0].getCUDATextureWrapper(),
            &canvas[1].getCUDATextureWrapper()
        };

        canvasCUDA[0]->cudaMapResources();
        canvasCUDA[0]->cudaCreateSurfaceObj();
        canvasCUDA[1]->cudaMapResources();
        canvasCUDA[1]->cudaCreateSurfaceObj();

        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        int frameIndex = 0;
        bool firstFrame = true;

        while ( isRunning )
        {
            int cur  = frameIndex & 1;
            int prev = cur ^ 1;


            camera_controller.cameraUpdate();
            benchmark.update();
            profiler.updateProfiler(benchmark.getCheckpointID(), visibleSpheresCount, notOccludedSpheresCount, visibleCylindersCount, notOccludedCylindersCount);

            if (window.getInput().isKeyDown(Key::T)) profiler.startSavingNextFrames(benchmark.getCheckpointID());

    
            Frustum frustum(camera);


            // Limpiar contadores en stream
            cudaMemsetAsync(frustumSpheresCounter, 0, sizeof(unsigned int), stream);
            cudaMemsetAsync(occlusionSpheresCounter, 0, sizeof(unsigned int), stream);
            cudaMemsetAsync(frustumCylindersCounter, 0, sizeof(unsigned int), stream);
            cudaMemsetAsync(occlusionCylindersCounter, 0, sizeof(unsigned int), stream);

            downsamplingDepthTexture(
                depthBufferPtr,
                canvasDepthDownsampleCUDA.getSurface(),
                SCR_WIDTH,
                SCR_HEIGHT,
                downsampleWorkGroupSizeX,
                downsampleWorkGroupSizeY,
                stream
            );


            
            cleaningScreen(
                canvasCUDA[cur]->getSurfaceObject(), 
                depthBufferPtr,
                pixelOwnershipBufferPtr,
                workGroupSizeXPerPixel,
                workGroupSizeYPerPixel,
                camera.getFar(),
                SCR_WIDTH,
                SCR_HEIGHT,
                stream
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
                frustumSpheresCounter,
                occlusionSpheresCounter,
                1,
                canvasCUDA[cur]->getSurfaceObject(),
                canvasDepthDownsampleCUDA.getTexture(),
                stream,
                workGroupSizeXPerSphere
            );

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
                frustumCylindersCounter,
                occlusionCylindersCounter,
                1,
                canvasCUDA[cur]->getSurfaceObject(),
                canvasDepthDownsampleCUDA.getTexture(),
                stream,
                workGroupSizeXPerCylinder
            );
            cudaEventRecord(evtReady[cur], stream);

            pixelCount(
                pixelOwnershipBufferPtr,
                visibilityFrameBufferPtr,
                workGroupSizeXPixelCount,
                workGroupSizeYPixelCount,
                SCR_WIDTH,
                SCR_HEIGHT,
                stream
            );

            if (!firstFrame) {
                
                // Copias asíncronas
                cudaMemcpyAsync(visibleSpheresCountPinned, frustumSpheresCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
                cudaMemcpyAsync(notOccludedSpheresCountPinned, occlusionSpheresCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
                cudaMemcpyAsync(visibleCylindersCountPinned, frustumCylindersCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
                cudaMemcpyAsync(notOccludedCylindersCountPinned, occlusionCylindersCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
                
                visibleSpheresCount = *visibleSpheresCountPinned;
                notOccludedSpheresCount = *notOccludedSpheresCountPinned;
                
                visibleCylindersCount = *visibleCylindersCountPinned;
                notOccludedCylindersCount = *notOccludedCylindersCountPinned;

                
                
                cudaEventSynchronize(evtReady[prev]);
                
                glBindFramebuffer(GL_READ_FRAMEBUFFER, canvas[prev].getFramebuffer().getId());
                isRunning = window.update();
                
            } else {
                // No hay prev listo en el primer frame
                firstFrame = false;
            }
            
            

            frameIndex++;
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
                canvas[cur].takeScreenshot(sshot_name);
            }

            if (window.getInput().isKeyDown(Key::F)) fbo1o2 = !fbo1o2; 

        }
        cudaEventDestroy(evtReady[0]);
        cudaEventDestroy(evtReady[1]);
        cudaStreamDestroy(stream);
        canvasCUDA[0]->cudaDestroySurfaceObj();
        canvasCUDA[0]->cudaUnmapResources();
        canvasCUDA[1]->cudaDestroySurfaceObj();
        canvasCUDA[1]->cudaUnmapResources();
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
    frustumSpheresAtomicCounterCUDAWrapper.cudaUnmapResources();
    occlusionSpheresAtomicCounterCUDAWrapper.cudaUnmapResources();
    frustumCylindersAtomicCounterCUDAWrapper.cudaUnmapResources();
    occlusionCylindersAtomicCounterCUDAWrapper.cudaUnmapResources();
    cudaFreeHost(visibleSpheresCountPinned);
    cudaFreeHost(notOccludedSpheresCountPinned);
    cudaFreeHost(visibleCylindersCountPinned);
    cudaFreeHost(notOccludedCylindersCountPinned);
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
    
    CanvasCUDA canvas[2] = {CanvasCUDA(GL_TEXTURE_2D, GL_RGBA8, SCR_WIDTH, SCR_HEIGHT),
                            CanvasCUDA(GL_TEXTURE_2D, GL_RGBA8, SCR_WIDTH, SCR_HEIGHT)};
    canvas[0].setFBO(GL_COLOR_ATTACHMENT0);
    canvas[0].setupDepthDataCUDA(SCR_WIDTH, SCR_HEIGHT);
    canvas[0].setupDepthDownsampleCUDA(downsampleLevel, SCR_WIDTH, SCR_HEIGHT);
    unsigned int* depthBufferPtr = canvas[0].getDepthDataCUDA().getDepthBuffer();
    DepthDownsampleCUDA& canvasDepthDownsampleCUDA = canvas[0].getDepthDownsampleDataCUDA();

    canvas[1].setFBO(GL_COLOR_ATTACHMENT0);
    
        auto cerr = cudaGetLastError();
    if (cerr != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(cerr));
    GLenum glerr = glGetError();
    if (glerr) std::cout << "GL error: 0x" << std::hex << glerr << std::dec << std::endl;


    std::vector<Sphere> spheres;
    std::vector<Cylinder> cylinders;
    getCompleteScene(spheres, sphere_count, cylinders, cylinder_count);

    sphere_count = spheres.size(); 
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
    spheres.clear();
    
    std::cout << "Cylinders Buffer" << std::endl;
    StorageBuffer cylinderBuffer(GL_SHADER_STORAGE_BUFFER, cylinder_count * sizeof(Cylinder), 7, 
        cylinders.data(), GL_STATIC_DRAW);
    StorageBufferCUDAWrapper cylinderBufferCUDAWrapper(cylinderBuffer);
    cylinderBufferCUDAWrapper.cudaMapResources();
    Cylinder * cylinderBufferPtr = cylinderBufferCUDAWrapper.getDevicePointer<Cylinder>();
    cylinders.clear();
    
    GLuint zero = 0;
    GLuint resetValue = 0;
    std::cout << "Frustum Spheres Atomic Counter" << std::endl;
    StorageBuffer frustumSpheresAtomicCounter(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint), 8, 
    &zero, GL_STATIC_DRAW);
    StorageBufferCUDAWrapper frustumSpheresAtomicCounterCUDAWrapper(frustumSpheresAtomicCounter);
    frustumSpheresAtomicCounterCUDAWrapper.cudaMapResources();
    unsigned int* frustumSpheresCounter = frustumSpheresAtomicCounterCUDAWrapper.getDevicePointer<unsigned int>();
    
    std::cout << "Frustum Cylinders Atomic Counter" << std::endl;
    StorageBuffer frustumCylindersAtomicCounter(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint), 8, 
    &zero, GL_STATIC_DRAW);
    StorageBufferCUDAWrapper frustumCylindersAtomicCounterCUDAWrapper(frustumCylindersAtomicCounter);
    frustumCylindersAtomicCounterCUDAWrapper.cudaMapResources();
    unsigned int* frustumCylindersCounter = frustumCylindersAtomicCounterCUDAWrapper.getDevicePointer<unsigned int>();
    
    // pinned memory for readback
    unsigned int* visibleSpheresCountPinned;
    unsigned int* visibleCylindersCountPinned;

    cudaHostAlloc(&visibleSpheresCountPinned, sizeof(unsigned int), cudaHostAllocDefault);
    cudaHostAlloc(&visibleCylindersCountPinned, sizeof(unsigned int), cudaHostAllocDefault);

    Benchmark benchmark(camera_controller, chkPoints);

    std::string base_path = "media/csv/fst_parallel_w_cyl/gpu_cull/"; 
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
    
    std::chrono::steady_clock::time_point endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = endTime - startTime;
    std::ofstream logFile("C:\\Users\\Sebatian\\Desktop\\XLIM\\Memoria_per_frame\\log.txt", std::ios::out);

    int count = 0;
    cudaGetDeviceCount(&count);

    for (int i = 0; i < count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "Device " << i << ": " << prop.name << std::endl;
        
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Memoria global: " << prop.totalGlobalMem / (1024.0 * 1024 * 1024) << " GB" << std::endl;
    }
    cudaSetDevice(0);
    
    try
    {
        bool fbo1o2 = false;
        bool isRunning = true;

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        cudaEvent_t evtReady[2];
        cudaEventCreateWithFlags(&evtReady[0], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&evtReady[1], cudaEventDisableTiming);

        TextureCUDAWrapper* canvasCUDA[2] = {
            &canvas[0].getCUDATextureWrapper(),
            &canvas[1].getCUDATextureWrapper()
        };

        canvasCUDA[0]->cudaMapResources();
        canvasCUDA[0]->cudaCreateSurfaceObj();
        canvasCUDA[1]->cudaMapResources();
        canvasCUDA[1]->cudaCreateSurfaceObj();

        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        int frameIndex = 0;
        bool firstFrame = true;

        while ( isRunning )
        {
            int cur  = frameIndex & 1;
            int prev = cur ^ 1;


            camera_controller.cameraUpdate();
            benchmark.update();
            profiler.updateProfiler(benchmark.getCheckpointID(), visibleSpheresCount, notOccludedSpheresCount, visibleCylindersCount, notOccludedCylindersCount);

            if (window.getInput().isKeyDown(Key::T)) profiler.startSavingNextFrames(benchmark.getCheckpointID());

    
            Frustum frustum(camera);


            // Limpiar contadores en stream
            cudaMemsetAsync(frustumSpheresCounter, 0, sizeof(unsigned int), stream);
            cudaMemsetAsync(frustumCylindersCounter, 0, sizeof(unsigned int), stream);

            cleaningScreenNoOcc(
                canvasCUDA[cur]->getSurfaceObject(), 
                depthBufferPtr,
                workGroupSizeXPerPixel,
                workGroupSizeYPerPixel,
                camera.getFar(),
                SCR_WIDTH,
                SCR_HEIGHT,
                stream
            );


            sphereRasterNoOcc(
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
                frustumSpheresCounter,
                1,
                canvasCUDA[cur]->getSurfaceObject(),
                stream,
                workGroupSizeXPerSphere
            );

            cylinderRasterNoOcc(
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
                frustumCylindersCounter,
                1,
                canvasCUDA[cur]->getSurfaceObject(),
                stream,
                workGroupSizeXPerCylinder
            );
            cudaEventRecord(evtReady[cur], stream);

            if (!firstFrame) {
                nvtxRangePushA("Async Copies and Updates");
                // Copias asíncronas
                cudaMemcpyAsync(visibleSpheresCountPinned, frustumSpheresCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
                cudaMemcpyAsync(visibleCylindersCountPinned, frustumCylindersCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
                
                visibleSpheresCount = *visibleSpheresCountPinned;
                
                visibleCylindersCount = *visibleCylindersCountPinned;
                
                cudaEventSynchronize(evtReady[prev]);
                // cudaStreamSynchronize(stream);
                nvtxRangePop();
                
                glBindFramebuffer(GL_READ_FRAMEBUFFER, canvas[prev].getFramebuffer().getId());
                isRunning = window.update();
                
            } else {
                // No hay prev listo en el primer frame
                firstFrame = false;
            }
            
            

            frameIndex++;
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
                canvas[cur].takeScreenshot(sshot_name);
            }

            if (window.getInput().isKeyDown(Key::F)) fbo1o2 = !fbo1o2; 

        }
        cudaEventDestroy(evtReady[0]);
        cudaEventDestroy(evtReady[1]);
        cudaStreamDestroy(stream);
        canvasCUDA[0]->cudaDestroySurfaceObj();
        canvasCUDA[0]->cudaUnmapResources();
        canvasCUDA[1]->cudaDestroySurfaceObj();
        canvasCUDA[1]->cudaUnmapResources();
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return ;
    }
    sphereBufferCUDAWrapper.cudaUnmapResources();
    cylinderBufferCUDAWrapper.cudaUnmapResources();
    frustumSpheresAtomicCounterCUDAWrapper.cudaUnmapResources();
    frustumCylindersAtomicCounterCUDAWrapper.cudaUnmapResources();
    cudaFreeHost(visibleSpheresCountPinned);
    cudaFreeHost(visibleCylindersCountPinned);
}