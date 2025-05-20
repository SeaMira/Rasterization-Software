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
bool withOcclusionCulling = false;

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

void mainWithOcclusionCulling(Camera& camera, AppOpenGL& window);
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

    ShaderProgram spheresShader("assets/shaders/standard_version/cpu_cull/spheres.vert", 
                                "assets/shaders/standard_version/cpu_cull/spheres.frag",
                                "assets/shaders/standard_version/cpu_cull/spheres.geom");
    spheresShader.linkProgram();
    
    ShaderProgram cylindersShader("assets/shaders/standard_version/cpu_cull/cylinders.vert", 
                                  "assets/shaders/standard_version/cpu_cull/cylinders.frag",
                                  "assets/shaders/standard_version/cpu_cull/cylinders.geom");
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
    
    std::vector<Sphere> visibleSpheres(sphere_count);
    
    StorageBuffer cylinderBuffer(GL_SHADER_STORAGE_BUFFER, cylinder_count * sizeof(Cylinder), 2, 
        cylinders.data(), GL_STATIC_DRAW);

    std::vector<Cylinder> visibleCylinders(cylinder_count);

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
            cullSimpleSpheres(spheres, visibleSpheres, frustum, visibleSpheresCount);
            sphereBuffer.bind();
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, visibleSpheresCount * sizeof(Sphere), visibleSpheres.data());
            sphereBuffer.unbind();

            spheresShader.use();
            spheresShader.setVec2I("screenResolution", screenResolution);
            setCameraUniforms(spheresShader, camera);
            glBindVertexArray(emptyVAO);
            glDrawArrays(GL_POINTS, 0, visibleSpheresCount);
            
            /// Cylinder drawing
            cullSimpleCylinders(cylinders, visibleCylinders, frustum, visibleCylindersCount);
            cylinderBuffer.bind();
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, visibleCylindersCount * sizeof(Cylinder), visibleCylinders.data());
            cylinderBuffer.unbind();
            
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