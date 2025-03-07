#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glad/glad.h>
#include <SDL3/SDL.h>
#include <vector>

#include <filesystem>
#include "molecule_loader/basic_loader.h"

#include "ux/input.h"
#include "ux/camera_controller.h"

#include "appSDLGL.h"
#include "vis/gl/frame_buffer.h"
#include "vis/gl/storage_buffer.h"
#include "vis/gl/texture.h"
#include "vis/canvas.h"
#include "vis/compute_shader_program.h"

using uint = unsigned int;

// Settings
const int SCR_WIDTH = 800;
const int SCR_HEIGHT = 600;

const int sphere_count = 128;

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
    
    std::filesystem::path path = "assets/molecules/1AGA.mmtf";
    ChemFilesLoader loader(path);
    std::vector<glm::vec4> positions = loader.getSphereInfo();
    // std::vector<glm::vec4> spheres(positions.begin(), positions.begin() + std::min(positions.size(), static_cast<size_t>(sphere_count)));
    std::vector<glm::vec4> spheres;
    for (int i = 0; i < sphere_count; i++)
    {
        spheres.push_back({(float)(i%100)*2.0f, (float)(i/100) * 2.0f, (float)(i%100)*2.0f, 1.0f});
    }
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
    
    // Calculating number of work groups (based on the number of threads and spheres)
    GLuint numGroupsX = (sphere_count + workGroupSizeX - 1) / workGroupSizeX;
    GLuint numGroupsY = 1;

    canvas.bindTexture(0);
    canvas.bindFBO();

    glm::ivec2 screenResolution(SCR_WIDTH, SCR_HEIGHT);
    float aspectRatio = ((float)SCR_WIDTH/(float)SCR_HEIGHT);
    try
    {
        bool isRunning = true;
        while ( isRunning )
        {
            camera_controller.cameraUpdate();

            // cleaning shader
            cleaningComputeShader.use();
            cleaningComputeShader.setVec2I("screenResolution", screenResolution);
            glDispatchCompute((SCR_WIDTH + workGroupSizeX - 1) / 16, (SCR_HEIGHT + workGroupSizeY - 1) / 16, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
            
            // bbox extraction shader
            bboxExtractionShader.use();
            bboxExtractionShader.setInt("sphereCount", sphere_count);
            bboxExtractionShader.setVec2I("screenResolution", screenResolution);
            bboxExtractionShader.setMat4("proj", camera.getProjection());
            bboxExtractionShader.setMat4("view", camera.getView());
            bboxExtractionShader.setVec3("up", camera.getUp());
            bboxExtractionShader.setVec3("front", camera.getFront());
            bboxExtractionShader.setVec3("cameraPos", camera.getPosition());
            bboxExtractionShader.setFloat("aspectRatio", aspectRatio);
            bboxExtractionShader.setFloat("fov", camera.getFov());
            glDispatchCompute(numGroupsX, numGroupsY, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
            
            // bbox intersection shader
            bboxIntersectionShader.use();
            bboxIntersectionShader.setInt("sphereCount", sphere_count);
            bboxIntersectionShader.setVec2I("screenResolution", screenResolution);
            bboxIntersectionShader.setMat4("proj", camera.getProjection());
            bboxIntersectionShader.setVec3("cameraPos", camera.getPosition());
            glDispatchCompute((SCR_WIDTH + workGroupSizeX - 1) / 16, (SCR_HEIGHT + workGroupSizeY - 1) / 16, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

            // Blit from framebuffer to default framebuffer (screen)
            glBindFramebuffer(GL_READ_FRAMEBUFFER, canvas.getFramebuffer().getId());

            isRunning = window.update();

            if (window.getInput().isKeyDown(Key::F10)) 
                canvas.takeScreenshot("off/scnd_parallel/test.bmp");
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }


    return 0;
}