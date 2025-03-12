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
int SCR_WIDTH = 800;
int SCR_HEIGHT = 600;
int sphere_count = 1024 * 1024;

std::string title = "First Parallel Version"; 

bool shown = true;

GLuint workGroupSizeX = 16;  // Deifining threads-per-group (X)
GLuint workGroupSizeY = 16;  // Deifining threads-per-group (Y)


int main(int argc, char* argv[]) 
{

    std::unordered_map<std::string, int*> scene_data = {
        {"Screen width", &SCR_WIDTH},
        {"Screen height", &SCR_HEIGHT},
        {"Sphere count", &sphere_count}
    };

    AppOpenGL window { title, SCR_WIDTH, SCR_HEIGHT, shown };

    Camera camera(SCR_WIDTH, SCR_HEIGHT);
    camera.SetPosition(.0f, .0f, .0f);
    CameraController camera_controller(window, camera);
    
    #if CPU_FRUSTUM_CULLING
        ComputeShader computeShader("assets/shaders/fst_parallel_attempt/fst_parallel.compute");
    #else
        ComputeShader computeShader("assets/shaders/fst_parallel_attempt/fst_parallel_culling.compute");
    #endif
    ComputeShader cleaningComputeShader("assets/shaders/fst_parallel_attempt/set_to_black.compute");

    Canvas canvas(GL_TEXTURE_2D, GL_RGBA8, SCR_WIDTH, SCR_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE);
    canvas.setFBO(GL_COLOR_ATTACHMENT0);
    
    if (!canvas.getFramebuffer().isComplete()) {
        throw std::runtime_error("Error: Incomplete Framebuffer.");
    }
    
    std::vector<glm::vec4> spheres = getScene(sphere_count);
    std::vector<std::pair<glm::vec3, glm::vec3>> chkPoints = getCheckpoints(sphere_count, spheres);
    
    #if CPU_FRUSTUM_CULLING
        int visibleSpheresCount = 0;
        std::vector<glm::vec4> visibleSpheres(spheres.size());
    #endif
    
    StorageBuffer sphereBuffer(GL_SHADER_STORAGE_BUFFER);
    sphereBuffer.generateBufferData(spheres.size() * sizeof(glm::vec4), 1, 
        spheres.data(), GL_STATIC_DRAW);
    sphereBuffer.unbind();
    
    std::vector<float> depth_b(SCR_WIDTH * SCR_HEIGHT, FLT_MAX);
    StorageBuffer depthBuffer(GL_SHADER_STORAGE_BUFFER);
    depthBuffer.generateBufferData(SCR_WIDTH * SCR_HEIGHT * sizeof(float), 2, depth_b.data(), GL_DYNAMIC_COPY);
    
    Benchmark benchmark(camera_controller, chkPoints);
    Profiler profiler(window, "media/off/fst_parallel/frame_times.off", "media/off/fst_parallel/process_times.off");
    
    window.setupSceneInfoGui("Scene Info", scene_data);
    window.setupCameraGui("Camera Info", &camera);
    window.setupInputInfoGui("General Input Info");
    window.setupBenchmarkInfoGui("Benchmark", &benchmark);
    
    // Calculating number of work groups (based on the number of threads and spheres)
    GLuint numGroupsX = (sphere_count + workGroupSizeX - 1) / workGroupSizeX;
    GLuint numGroupsY = 1;
    depthBuffer.unbind();

    canvas.bindTexture(0);
    canvas.bindFBO();

    glm::ivec2 screenResolution(SCR_WIDTH, SCR_HEIGHT);
    try
    {
        bool isRunning = true;
        while ( isRunning )
        {
            camera_controller.cameraUpdate();
            benchmark.update();
            profiler.updateProfiler();

            if (window.getInput().isKeyDown(Key::T)) profiler.startSavingNextFrames(benchmark.getCheckpointID());

            cleaningComputeShader.use();
            cleaningComputeShader.setFloat("far", camera.getFar());
            cleaningComputeShader.setVec2I("screenResolution", screenResolution);
            glDispatchCompute((SCR_WIDTH + workGroupSizeX - 1) / 16, (SCR_HEIGHT + workGroupSizeY - 1) / 16, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);


            computeShader.use();
            Frustum frustum(camera);
            #if CPU_FRUSTUM_CULLING
                cullSpheres(spheres, visibleSpheres, frustum, visibleSpheresCount);
                sphereBuffer.bind();
                glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, visibleSpheresCount * sizeof(glm::vec4), visibleSpheres.data());
                sphereBuffer.unbind();
                numGroupsX = (visibleSpheresCount + workGroupSizeX - 1) / workGroupSizeX;
                
                computeShader.setInt("sphereCount", visibleSpheresCount); // visible sphere count given
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
            computeShader.setVec3("cameraPos", camera.getPosition());

            glDispatchCompute(numGroupsX, numGroupsY, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

            
            // Blit from framebuffer to default framebuffer (screen)
            glBindFramebuffer(GL_READ_FRAMEBUFFER, canvas.getFramebuffer().getId());
            
            isRunning = window.update();
            
            if (window.getInput().isKeyDown(Key::F10)) 
            {
                std::string sshot_name  = "media/img/fst_parallel/frame_" + std::to_string(benchmark.getCheckpointID()) + ".bmp";
                canvas.takeScreenshot(sshot_name);
            }

        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }


    return 0;
}