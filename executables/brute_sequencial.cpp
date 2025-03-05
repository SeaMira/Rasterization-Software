#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glad/glad.h>
#include <SDL3/SDL.h>
#include <vector>
#include <cmath>
#include <string>

#include <filesystem>
#include "molecule_loader/basic_loader.h"

#include "utils/math_defines.h"
#include "utils/sequential/aux_functions.h"

#include "ux/input.h"
#include "ux/camera_controller.h"
#include "ux/cinematic/benchmark.h"
#include "ux/profiler/profiler.h"

#include "vis/window.h"
#include "vis/gl/frame_buffer.h"
#include "vis/gl/texture.h"
#include "vis/canvas.h"
#include "vis/compute_shader_program.h"

// Settings
const int SCR_WIDTH = 800;
const int SCR_HEIGHT = 600;

const int sphere_count = 126;

std::string title = "Brute Sequential Method"; 

bool shown = true;

// benchmark settings
int spheresGridWidth = 100;
int interleaveW = 5;
int interleaveH = 5;

int interleaveAngle = 10;
int interleaveZ = 10;
int interleaveY = 10;


std::vector<std::pair<glm::vec3, glm::vec3>> benchmark1_structured_grid(int spheresGridWidth, int interleaveW, int interleaveH, int interleaveZ);
std::vector<std::pair<glm::vec3, glm::vec3>> benchmark2_loaded_molecules(std::vector<glm::vec4>& spheres, int interleaveAngle, int interleaveZ, int interleaveY);

void renderFrame(std::vector<uint32_t>& framebuffer, 
    std::vector<float>& depthBuffer, std::vector<glm::vec4>& spheres,
    Camera& cam) 
{
    const glm::mat4& proj = cam.getProjection();
    const glm::mat4& view = cam.getView();
    const glm::vec3& up = cam.getUp();
    // const glm::vec3& right = cam.getRight();
    const glm::vec3& front = cam.getFront();
    const glm::vec3& camPos = cam.getPosition();
    // const glm::vec2 resol = glm::vec2(SCR_WIDTH, SCR_HEIGHT);
    const float fov = cam.getFov();
    float aspectRatio = ((float)SCR_WIDTH/(float)SCR_HEIGHT);
    
    for (const auto& sphere : spheres) 
    {
        drawSphere(proj, view, up, front, camPos, SCR_WIDTH, SCR_HEIGHT, 
            fov, aspectRatio, sphere, 
            framebuffer, depthBuffer);
    }
}

void drawSDL(SDL_Renderer* renderer, SDL_Texture* texture, std::vector<uint32_t>& framebuffer) {
    SDL_UpdateTexture(texture, nullptr, framebuffer.data(), SCR_WIDTH * sizeof(uint32_t));
    SDL_RenderClear(renderer);
    SDL_RenderTexture(renderer, texture, nullptr, nullptr);
    SDL_RenderPresent(renderer);
}

int main(int argc, char* argv[]) 
{
    Window window { title, SCR_WIDTH, SCR_HEIGHT, shown };
    Camera camera(SCR_WIDTH, SCR_HEIGHT);
    camera.SetPosition(.0f, .0f, .0f);
    CameraController camera_controller(window, camera);
    
    // benchmark settings
    // std::vector<std::pair<glm::vec3, glm::vec3>> chkPoints = benchmark1_structured_grid(spheresGridWidth, interleaveW, interleaveH, interleaveZ);
    
    // Benchmark benchmark(camera_controller, chkPoints);
    // std::vector<glm::vec4> spheres;
    // for (int i = 0; i < sphere_count; i++)
    // {
    //     spheres.push_back({(float)(i%spheresGridWidth)*2.0f, (float)(i/spheresGridWidth) * 2.0f, (float)(i%spheresGridWidth)*2.0f, 1.0f});
    // }
    
    std::vector<uint32_t> framebuffer(SCR_WIDTH * SCR_HEIGHT);
    std::vector<float> depthBuffer(SCR_WIDTH * SCR_HEIGHT, FLT_MAX);
    
    SDL_Renderer* renderer = SDL_CreateRenderer(window.getHandle(), nullptr);
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, SCR_WIDTH, SCR_HEIGHT);
   
    std::filesystem::path path = "molecules/1AGA.mmtf";
    ChemFilesLoader loader(path);
    std::vector<glm::vec4> positions = loader.getSphereInfo();
    std::vector<glm::vec4> spheres(positions.begin(), positions.begin() + std::min(positions.size(), static_cast<size_t>(sphere_count)));
    std::vector<std::pair<glm::vec3, glm::vec3>> chkPoints = benchmark2_loaded_molecules(spheres, interleaveAngle, interleaveZ, interleaveY);
    
    Benchmark benchmark(camera_controller, chkPoints);
    Profiler profiler(window, "off/seq/frame_times.off", "off/seq/process_times.off");
    try
    {
        bool isRunning = true;
        while ( isRunning )
        {
            
            camera_controller.cameraUpdate();
            benchmark.update();

            profiler.updateProfiler();
            if (window.getInput().isKeyDown(Key::T)) profiler.startSavingNextFrames(benchmark.getCheckpointID());

            std::fill(framebuffer.begin(), framebuffer.end(), 0xFFFFFF00);
            std::fill(depthBuffer.begin(), depthBuffer.end(), FLT_MAX);
            
            renderFrame(framebuffer, depthBuffer, spheres, camera);
            drawSDL(renderer, texture, framebuffer);
            if (window.getInput().isKeyDown(Key::F10)) 
            {
                SDL_Surface* sshot = SDL_RenderReadPixels(renderer, nullptr);
                bool sshot_saved = SDL_SaveBMP(sshot, "off/seq/test.bmp");
                if (sshot_saved) std::cout << "sshot saved." << std::endl;
                else std::cout << "sshot couldnt be saved." << std::endl;
                SDL_DestroySurface(sshot);
            }

            isRunning = window.update();
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        SDL_DestroyTexture(texture);
        SDL_DestroyRenderer(renderer);
        return 1;
    }
    
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);

    return 0;
}


std::vector<std::pair<glm::vec3, glm::vec3>> benchmark1_structured_grid(int spheresGridWidth, int interleaveW, int interleaveH, int interleaveZ)
{
    std::vector<std::pair<glm::vec3, glm::vec3>> chkPoints;

    float h_delta = (sphere_count/spheresGridWidth) * 2.0f / interleaveH;
    float w_delta = (float)(spheresGridWidth / interleaveW) * 2.0f;
    for (int i = 0; i < interleaveW; i++)
    {
        for (int j = 0; j < interleaveH; j++)
        {
            chkPoints.push_back(
                {
                    glm::vec3(w_delta*(float)i + 1.0f, h_delta*(float)j, w_delta*(float)i - 3.0f), 
                    glm::vec3(w_delta*(float)i, h_delta*(float)j, w_delta*(float)i)
                }
            );
        }
    }

    for (int i = 0; i < interleaveW; i++)
    {
        for (int j = 0; j < interleaveZ; j++)
        {
            chkPoints.push_back(
                {
                    glm::vec3(w_delta*(float)i + (float)j + 1.0f, h_delta*interleaveH/2.0f, w_delta*(float)i - (float)j - 1.0f), 
                    glm::vec3(w_delta*(float)i, h_delta*interleaveH/2.0f, w_delta*(float)i)
                }
            );
        }
    }
    return chkPoints;
}


std::vector<std::pair<glm::vec3, glm::vec3>> benchmark2_loaded_molecules(std::vector<glm::vec4>& spheres, int interleaveAngle, int interleaveZ, int interleaveY)
{
    float x_max = FLT_MIN;
    float y_max = FLT_MIN, y_min = FLT_MAX;
    float z_max = FLT_MIN;
    glm::vec3 mass_center(0.0f);

    for (auto& sphere : spheres)
    {
        mass_center += glm::vec3(sphere);
        if (sphere.y < y_min) y_min = sphere.y;
        if (sphere.y > y_max) y_max = sphere.y;
        if (abs(sphere.x) > x_max) x_max = abs(sphere.x);
        if (abs(sphere.z) > z_max) z_max = abs(sphere.z);
    }
    mass_center /= spheres.size();

    float d_theta = 360.0f/(float)interleaveAngle;
    float radius = sqrt(x_max*x_max + z_max*z_max);
    float d_radius = radius/(float) interleaveZ;
    float d_height = (y_max - y_min) / (float) interleaveY;

    std::vector<std::pair<glm::vec3, glm::vec3>> chkPoints;

    for (int i = 0; i < interleaveAngle; i++)
    {
        for (int j = 1; j < interleaveZ; j++)
        {
            for (int k = 0; k <= interleaveY; k++)
            {
                float d = (float)j * d_radius;
                chkPoints.push_back(
                    {
                        mass_center +
                        glm::vec3(d*cos(glm::radians((float)i*d_theta)), 
                        y_min + (float)k*d_height, 
                        d*sin(glm::radians((float)i*d_theta))), 
                        glm::vec3(mass_center.x, y_min + (float)k*d_height, mass_center.z)
                    }
                );
            }
        }
    }

    return chkPoints;
}