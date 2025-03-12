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
int SCR_WIDTH = 1000;
int SCR_HEIGHT = 1000;
int sphere_count = 1024 * 1024;

std::string title = "Second Parallel Version"; 

bool shown = true;

GLuint workGroupSizeX = 16;  // Deifining threads-per-group (X)
GLuint workGroupSizeY = 16;  // Deifining threads-per-group (Y)

struct SphereBillboard 
{
    glm::vec4 cameraSpaceSphPosR;   // 16b - camera space position of the sphere and radius
    glm::vec4 upRightCornerMaxX;     // 16b - up right corner of the billboard in camera space and the max X coordinate in screen space 
    glm::vec4 upLeftCornerMaxY;      // 16b - up left corner of the billboard in camera space and the max Y coordinate in screen space
    glm::vec4 downRightCornerMinX;   // 16b - down right corner of the billboard in camera space and the min X coordinate in screen space
    glm::vec4 downLeftCornerMinY;    // 16b - down left corner of the billboard in camera space and the min Y coordinate in screen space
    // 80b
}; 

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
    
    ComputeShader bboxExtractionShader("assets/shaders/snd_parallel_attempt/bbox_extraction.compute");
    ComputeShader bboxIntersectionShader("assets/shaders/snd_parallel_attempt/bbox_intersect.compute");
    ComputeShader cleaningComputeShader("assets/shaders/snd_parallel_attempt/set_to_black.compute");

    Canvas canvas(GL_TEXTURE_2D, GL_RGBA8, SCR_WIDTH, SCR_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE);
    canvas.setFBO(GL_COLOR_ATTACHMENT0);
    
    if (!canvas.getFramebuffer().isComplete()) 
    {
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
    
    // std::vector<float> depth_b(SCR_WIDTH * SCR_HEIGHT, FLT_MAX);
    // StorageBuffer depthBuffer(GL_SHADER_STORAGE_BUFFER);
    // depthBuffer.generateBufferData(SCR_WIDTH * SCR_HEIGHT * sizeof(float), 2, depth_b.data(), GL_DYNAMIC_COPY);
    // depthBuffer.unbind();
    
    std::vector<SphereBillboard> billboards(spheres.size());
    StorageBuffer sphereBillboardBuffer(GL_SHADER_STORAGE_BUFFER);
    sphereBillboardBuffer.generateBufferData(spheres.size() * sizeof(SphereBillboard), 2, billboards.data(), GL_DYNAMIC_COPY);
    sphereBillboardBuffer.unbind();
    
    Benchmark benchmark(camera_controller, chkPoints);
    Profiler profiler(window, "media/off/fst_parallel/frame_times.off", "media/off/fst_parallel/process_times.off");
    
    window.setupSceneInfoGui("Scene Info", scene_data);
    window.setupCameraGui("Camera Info", &camera);
    window.setupInputInfoGui("General Input Info");
    window.setupBenchmarkInfoGui("Benchmark", &benchmark);

    // Calculating number of work groups (based on the number of threads and spheres)
    GLuint numGroupsX = (sphere_count + workGroupSizeX - 1) / workGroupSizeX;
    GLuint numGroupsY = 1;

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
            
            #if CPU_FRUSTUM_CULLING
                Frustum frustum(camera);
                cullSpheres(spheres, visibleSpheres, frustum, visibleSpheresCount);
                sphereBuffer.bind();
                glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, visibleSpheresCount * sizeof(glm::vec4), visibleSpheres.data());
                sphereBuffer.unbind();
                numGroupsX = (visibleSpheresCount + workGroupSizeX - 1) / workGroupSizeX;
            #endif

            if (window.getInput().isKeyDown(Key::T)) profiler.startSavingNextFrames(benchmark.getCheckpointID());

            // cleaning shader
            cleaningComputeShader.use();
            cleaningComputeShader.setVec2I("screenResolution", screenResolution);
            glDispatchCompute((SCR_WIDTH + workGroupSizeX - 1) / 16, (SCR_HEIGHT + workGroupSizeY - 1) / 16, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
            
            // bbox extraction shader
            bboxExtractionShader.use();
            #if CPU_FRUSTUM_CULLING
                bboxExtractionShader.setInt("sphereCount", visibleSpheresCount); // visible sphere count given
            #else
                bboxExtractionShader.setInt("sphereCount", sphere_count);
            #endif
            bboxExtractionShader.setVec2I("screenResolution", screenResolution);
            bboxExtractionShader.setMat4("proj", camera.getProjection());
            bboxExtractionShader.setMat4("view", camera.getView());
            bboxExtractionShader.setVec3("up", camera.getUp());
            bboxExtractionShader.setVec3("front", camera.getFront());
            bboxExtractionShader.setVec3("cameraPos", camera.getPosition());
            glDispatchCompute(numGroupsX, numGroupsY, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
            
            // bbox intersection shader
            bboxIntersectionShader.use();
            #if CPU_FRUSTUM_CULLING
                bboxIntersectionShader.setInt("sphereCount", visibleSpheresCount); // visible sphere count given
            #else
                bboxIntersectionShader.setInt("sphereCount", sphere_count);
            #endif
            bboxIntersectionShader.setFloat("far", camera.getFar());
            bboxIntersectionShader.setVec2I("screenResolution", screenResolution);
            bboxIntersectionShader.setMat4("proj", camera.getProjection());
            bboxIntersectionShader.setVec3("cameraPos", camera.getPosition());
            glDispatchCompute((SCR_WIDTH + workGroupSizeX - 1) / 16, (SCR_HEIGHT + workGroupSizeY - 1) / 16, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

            // Blit from framebuffer to default framebuffer (screen)
            glBindFramebuffer(GL_READ_FRAMEBUFFER, canvas.getFramebuffer().getId());

            isRunning = window.update();

            if (window.getInput().isKeyDown(Key::F10)) 
            {
                std::string sshot_name  = "media/img/scnd_parallel/frame_" + std::to_string(benchmark.getCheckpointID()) + ".bmp";
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