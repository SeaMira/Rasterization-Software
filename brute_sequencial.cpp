#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glad/glad.h>
#include <SDL3/SDL.h>
#include <vector>

#include <filesystem>
#include "molecule_loader/basic_loader.h"

#include "utils/math_defines.h"
#include "utils/sequential/aux_functions.h"

#include "ux/input.h"
#include "ux/camera_controller.h"

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

// bool isInsideEllipse(int x, int y, ProjectionResult& elipse) 
// {
//     // int ndcX = x;
//     // int ndcY = y;
//     float ndcX = (2.0f * x) / SCR_WIDTH - 1.0f;
//     float ndcY = (2.0f * y) / SCR_HEIGHT - 1.0f;
//     float F = elipse.a * ndcX * ndcX + 
//               elipse.b * ndcX * ndcY + 
//               elipse.c * ndcY * ndcY + 
//               elipse.d * ndcX + 
//               elipse.e * ndcY + 
//               elipse.f;

//     // std::cout << F << std::endl;
//     return 0.0f <= F && F <= 1.0f ; // Margen de error
// }


// ProjectionResult projectSphere( /* sphere */ glm::vec4 sph, 
//     /* camera matrix */ glm::mat4 cam,
//     /* projection    */ float fle )
// {
// // transform to camera space	
// glm::vec3  sph3(sph[0], sph[1], sph[2]);
// glm::vec3  o = glm::vec3(cam*glm::vec4(sph3, 1.0f));

// float r2 = sph[3]*sph[3];
// float z2 = o[2]*o[2];	
// float l2 = dot(o,o);

// float r2z2 = r2-z2;
// float area = -3.141593*fle*fle*r2*sqrt(abs((l2-r2)/(r2z2)))/(r2z2 + 1e-12);

// // axis
// glm::vec2 axa = fle*sqrt(-r2*(r2-l2)/((l2-z2)*(r2z2)*(r2z2) + 1e-12f)) * glm::vec2( o[0],o[1]);
// glm::vec2 axb = fle*sqrt(-r2/((l2-z2)*(r2z2) + 1e-6f))*glm::vec2(-o[1],o[0]);

// // center
// glm::vec2  cen = fle*o.z*glm::vec2(o[0], o[1])/(z2-r2);


// return { area, cen, axa, axb, 
// /* a */ r2 - o.y*o.y - z2,
// /* b */ 2.0f*o.x*o.y,
// /* c */ r2 - o.x*o.x - z2,
// /* d */ -2.0f*o.x*o.z*fle,
// /* e */ -2.0f*o.y*o.z*fle,
// /* f */ (r2-l2+z2)*fle*fle };

// }

void renderFrame(std::vector<uint32_t>& framebuffer, 
    std::vector<float>& depthBuffer, std::vector<std::pair<glm::vec3, float>>& spheres,
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
    
    std::vector<uint32_t> framebuffer(SCR_WIDTH * SCR_HEIGHT);
    std::vector<float> depthBuffer(SCR_WIDTH * SCR_HEIGHT, FLT_MAX);
    
    SDL_Renderer* renderer = SDL_CreateRenderer(window.getHandle(), nullptr);
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, SCR_WIDTH, SCR_HEIGHT);
   
    std::filesystem::path path = "molecules/1AGA.mmtf";
    ChemFilesLoader loader(path);
    std::vector<std::pair<glm::vec3, float>> positions = loader.getSphereInfo();
    std::vector<std::pair<glm::vec3, float>> spheres(positions.begin(), positions.begin() + std::min(positions.size(), static_cast<size_t>(sphere_count)));

    try
    {
        bool isRunning = true;
        while ( isRunning )
        {
            
            camera_controller.keyBoardAction();
            camera_controller.mouseAction();

            std::fill(framebuffer.begin(), framebuffer.end(), 0xFFFFFF00);
            std::fill(depthBuffer.begin(), depthBuffer.end(), FLT_MAX);
            
            renderFrame(framebuffer, depthBuffer, spheres, camera);
            drawSDL(renderer, texture, framebuffer);

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