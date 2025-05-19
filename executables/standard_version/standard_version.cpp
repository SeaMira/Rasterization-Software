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
int sphere_count = 64;

std::string title = "Standard OpenGL Version: Spheres and Cylinders"; 

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

void mainWithoutOcclusionCulling(Camera& camera, AppOpenGL& window);

int main(int argc, char* argv[]) 
{
    AppOpenGL window { title, SCR_WIDTH, SCR_HEIGHT, shown };

    Camera camera(SCR_WIDTH, SCR_HEIGHT);
    camera.SetPosition(.0f, .0f, .0f);

    mainWithoutOcclusionCulling(camera, window);

    return 0;
}


void mainWithoutOcclusionCulling(Camera& camera, AppOpenGL& window)
{
    std::unordered_map<std::string, int*> scene_data = {
        {"Screen width", &SCR_WIDTH},
        {"Screen height", &SCR_HEIGHT},
        {"Sphere count", &sphere_count},
        {"Spheres On Frustum", &visibleSpheresCount},
        {"Drawn spheres", &notOccludedSpheresCount}
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

    ComputeShader spheresCullingShader("assets/shaders/standard_version/spheresCulling.compute");
    ShaderProgram spheresShader("assets/shaders/standard_version/spheres.vert", 
                                "assets/shaders/standard_version/spheres.frag",
                                "assets/shaders/standard_version/spheres.geom");
    spheresShader.linkProgram();

    std::vector<Sphere> spheres = getScene(sphere_count);
    sphere_count = spheres.size(); 

    std::vector<std::pair<glm::vec3, glm::vec3>> chkPoints = getCheckpoints(sphere_count, spheres);

    StorageBuffer sphereBuffer(GL_SHADER_STORAGE_BUFFER, sphere_count * sizeof(Sphere), 0, 
        spheres.data(), GL_STATIC_DRAW);
    std::cout << "Spheres buffer " << std::endl;
    
    std::vector<GLuint> visibleSpheres(sphere_count, 0);
    StorageBuffer visibleSpheresBuffer(GL_SHADER_STORAGE_BUFFER, sphere_count * sizeof(GLuint), 1, 
        visibleSpheres.data(), GL_STATIC_DRAW);
    std::cout << "Sphere indexes buffer " << std::endl;

    GLuint zero = 0;
    GLuint resetValue = 0;
    StorageBuffer visibleSpheresAtomicCounter(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint), 2, 
    &zero, GL_STATIC_DRAW);

    GLuint emptyVAO;
    glCreateVertexArrays(1, &emptyVAO);

    Benchmark benchmark(camera_controller, chkPoints);

    std::string base_path = "media/csv/standard_version/"; 
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

            visibleSpheresAtomicCounter.bind();
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &visibleSpheresCount);
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
            visibleSpheresAtomicCounter.unbind();

            spheresCullingShader.use();
            spheresCullingShader.setInt("sphereCount", sphere_count);
            setFrustumUniforms(spheresCullingShader, frustum);
            glDispatchCompute(numGroupsXSpheres, numGroupsY, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

            spheresShader.use();
            spheresShader.setVec2I("screenResolution", screenResolution);
            setCameraUniforms(spheresShader, camera);
            glBindVertexArray(emptyVAO);
            glDrawArraysInstanced(GL_POINTS, 0, 1, visibleSpheresCount);
            
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