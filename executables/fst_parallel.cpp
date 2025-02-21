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

#include "vis/window.h"
#include "vis/gl/frame_buffer.h"
#include "vis/gl/storage_buffer.h"
#include "vis/gl/texture.h"
#include "vis/canvas.h"
#include "vis/compute_shader_program.h"

using uint = unsigned int;

// Settings
const int SCR_WIDTH = 800;
const int SCR_HEIGHT = 600;

const int sphere_count = 20;

std::string title = "First Parallel Version"; 

bool shown = true;

GLuint workGroupSizeX = 16;  // Deifining threads-per-group (X)
GLuint workGroupSizeY = 16;  // Deifining threads-per-group (Y)


int main(int argc, char* argv[]) 
{
    Window window { title, SCR_WIDTH, SCR_HEIGHT, shown };
    Camera camera(SCR_WIDTH, SCR_HEIGHT);
    camera.SetPosition(.0f, .0f, .0f);
    CameraController camera_controller(window, camera);
    
    ComputeShader computeShader("shaders/fst_parallel.compute");
    ComputeShader cleaningComputeShader("shaders/set_to_black.compute");

    Canvas canvas(GL_TEXTURE_2D, GL_RGBA8, SCR_WIDTH, SCR_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE);
    canvas.setFBO(GL_COLOR_ATTACHMENT0);
    
    if (!canvas.getFramebuffer().isComplete()) {
        throw std::runtime_error("Error: Incomplete Framebuffer.");
    }
    
    std::filesystem::path path = "molecules/1AGA.mmtf";
    ChemFilesLoader loader(path);
    std::vector<glm::vec4> positions = loader.getSphereInfo();
    std::vector<glm::vec4> spheres(positions.begin(), positions.begin() + std::min(positions.size(), static_cast<size_t>(sphere_count)));

    StorageBuffer sphereBuffer(GL_SHADER_STORAGE_BUFFER);
    sphereBuffer.generateBufferData(spheres.size() * sizeof(glm::vec4), 1, 
        spheres.data(), GL_STATIC_DRAW);
    sphereBuffer.unbind();
    
    StorageBuffer depthBuffer(GL_SHADER_STORAGE_BUFFER);
    depthBuffer.generateBufferData(SCR_WIDTH * SCR_HEIGHT * sizeof(int), 2);
    // Calculating number of work groups (based on the number of threads and spheres)
    GLuint numGroupsX = (sphere_count + workGroupSizeX - 1) / workGroupSizeX;
    GLuint numGroupsY = 1;
    depthBuffer.unbind();

    canvas.bindTexture(0);
    canvas.bindFBO();

    glm::ivec2 screenResolution(SCR_WIDTH, SCR_HEIGHT);
    float aspectRatio = ((float)SCR_WIDTH/(float)SCR_HEIGHT);
    try
    {
        bool isRunning = true;
        while ( isRunning )
        {
            camera_controller.keyBoardAction();
            camera_controller.mouseAction();

            cleaningComputeShader.use();
            glDispatchCompute((SCR_WIDTH + workGroupSizeX - 1) / 16, (SCR_HEIGHT + workGroupSizeY - 1) / 16, 1);

            computeShader.use();
            computeShader.setInt("sphereCount", sphere_count);
            computeShader.setVec2I("screenResolution", screenResolution);
            computeShader.setMat4("proj", camera.getProjection());
            computeShader.setMat4("view", camera.getView());
            computeShader.setVec3("up", camera.getUp());
            computeShader.setVec3("front", camera.getFront());
            computeShader.setVec3("cameraPos", camera.getPosition());
            computeShader.setFloat("aspectRatio", aspectRatio);
            computeShader.setFloat("fov", camera.getFov());

            glDispatchCompute(numGroupsX, numGroupsY, 1);
            glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
            // Blit from framebuffer to default framebuffer (screen)
            glBindFramebuffer(GL_READ_FRAMEBUFFER, canvas.getFramebuffer().getId());
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
            glBlitFramebuffer(0, 0, SCR_WIDTH, SCR_HEIGHT, 0, 0, SCR_WIDTH, SCR_HEIGHT, GL_COLOR_BUFFER_BIT, GL_NEAREST);

            isRunning = window.update();
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}