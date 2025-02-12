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

const int sphere_count = 1;

std::string title = "Brute sequencial method"; 

bool shown = true;


// Convierte coordenadas de mundo a coordenadas de pantalla
void worldToScreen(float wx, float wy, float wz, int& sx, int& sy, float& depth, Camera& cam) 
{
    float scale = tan(cam.getFov() * 0.5 * _PI / 180.0);  // Conversión de FOV a radianes
    float aspectRatio = (float)SCR_WIDTH / SCR_HEIGHT;  // Relación de aspecto

    // Proyección en perspectiva adaptada al sistema con UP = (0, 0, 1)
    sx = (int)((wx / (scale * aspectRatio * wy)) * SCR_WIDTH * 0.5f + SCR_WIDTH * 0.5f);
    sy = (int)((-wz / (scale * wy)) * SCR_HEIGHT * 0.5f + SCR_HEIGHT * 0.5f);
    
    // La profundidad se mide respecto al nuevo eje de profundidad (wy en vez de wz)
    depth = wy;
}

uint32_t depthToColor(float depth, float minDepth, float maxDepth) 
{
    // Normalizar la profundidad a un rango de 0 a 1
    float normalizedDepth = (depth - minDepth) / (maxDepth - minDepth);
    normalizedDepth = glm::clamp(normalizedDepth, 0.0f, 1.0f);

    // Mapear el valor normalizado a un color en escala de grises
    uint8_t colorValue = static_cast<uint8_t>(normalizedDepth * 255.0f);
    uint32_t color = (colorValue << 16) | (colorValue << 8) | colorValue;
    // std::cout << "color " << color << std::endl;
    return color; // RGB
}

void renderFrame(Camera& cam, std::vector<uint32_t>& framebuffer, 
    std::vector<float>& depthBuffer, std::vector<std::pair<glm::vec3, float>>& spheres) 
{
    float minDepth = 5000.0f, maxDepth = 0.0f;
    for (const auto& sphere : spheres) {
        int sx, sy;
        float depth;
        
        glm::vec3 spherePos = sphere.first;
        // std::cout << "Sphere position: " << spherePos.x << " " << spherePos.y << " " << spherePos.z << std::endl;
        float radius = sphere.second;
        // std::cout << "Sphere radius: " << radius << std::endl;
        
        // Convertir centro de la esfera a coordenadas de pantalla
        worldToScreen(spherePos.x, spherePos.y, spherePos.z, sx, sy, depth, cam);
        // std::cout << "Pixel: " << sx << " " << sy << " Depth: " << depth << std::endl;
        
        if (depth <= 0) continue; // Descartar esferas detrás de la cámara
        
        // Convertir radio de la esfera a píxeles en pantalla
        int rScreen;
        float dx = radius / depth;
        worldToScreen(spherePos.x + dx, spherePos.y, spherePos.z, rScreen, sy, depth, cam);
        int radiusPixels = abs(rScreen - sx);
        // std::cout << "Radius Pixel: " << radiusPixels << std::endl;

        // Rasterizar la esfera en pantalla
        for (int py = -radiusPixels; py <= radiusPixels; py++) 
        {
            for (int px = -radiusPixels; px <= radiusPixels; px++) 
            {
                int x = sx + px;
                int y = sy + py;
                
                if (x < 0 || x >= SCR_WIDTH || y < 0 || y >= SCR_HEIGHT) continue;

                // Checar si el píxel está dentro del círculo
                if (px * px + py * py > radiusPixels * radiusPixels) continue;

                int index = y * SCR_WIDTH + x;
                // std::cout << "Index: " << index << " from " << x << ", " << y << std::endl;
                
                // Calcular profundidad del píxel basado en la ecuación de la esfera
                float dy = sqrt(radius * radius - (px * px + py * py) * depth * depth / (radiusPixels * radiusPixels));
                float pixelDepth = spherePos.y - dy;
                if (pixelDepth > maxDepth) maxDepth = pixelDepth;
                if (pixelDepth < minDepth) minDepth = pixelDepth;
                // Actualizar buffers si el píxel está más cerca
                if (pixelDepth < depthBuffer[index]) {
                    depthBuffer[index] = pixelDepth;
                    framebuffer[index] = depthToColor(pixelDepth, minDepth, maxDepth);
                }
            }
        }
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
    camera.SetPosition(3.0f, 0.0f, 0.0f);
    CameraController camera_controller(window, camera);
    
    std::vector<uint32_t> framebuffer(SCR_WIDTH * SCR_HEIGHT);
    std::vector<float> depthBuffer(SCR_WIDTH * SCR_HEIGHT, FLT_MAX);
    
    SDL_Renderer* renderer = SDL_CreateRenderer(window.getHandle(), nullptr);
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, SCR_WIDTH, SCR_HEIGHT);
   
    std::filesystem::path path = "molecules/1AGA.mmtf";
    ChemFilesLoader loader(path);
    std::vector<std::pair<glm::vec3, float>> positions = loader.getSphereInfo();
    // std::vector<std::pair<glm::vec3, float>> spheres(positions.begin(), positions.begin() + std::min(positions.size(), static_cast<size_t>(sphere_count)));
    std::vector<std::pair<glm::vec3, float>> spheres {{glm::vec3(0.0f, 3.0f, 0.0f), 1.0f}, 
        {glm::vec3(1.0f, 3.0f, 0.0f), 2.0f},
        {glm::vec3(-1.0f, 3.0f, 0.0f), 2.0f},
        {glm::vec3(0.5f, 3.0f, 0.0f), 2.0f},
        {glm::vec3(1.0f, 3.0f, 0.0f), 1.0f},
        {glm::vec3(-1.0f, 3.0f, 0.0f), 1.0f},
        {glm::vec3(0.5f, 3.0f, 0.0f), 1.0f},
        {glm::vec3(-0.5f, 3.0f, 0.0f), 1.0f}};
    
    try
    {
        bool isRunning = true;
        while ( isRunning )
        {
            
            // if (window.getInput().isKeyDown(Key::G)) show_grid = !show_grid;
            // if (window.getInput().isKeyDown(Key::H)) show_axes = !show_axes;
            // if (window.getInput().isKeyDown(Key::J)) show_num = !show_num;
            // if (window.getInput().isKeyDown(Key::F)) std::cout << 1/window.getInput().deltaTime << std::endl;
            
            // if (show_fps) std::cout << 1/window.getInput().deltaTime << std::endl;
            
            camera_controller.keyBoardAction();
            camera_controller.mouseAction();

            std::fill(framebuffer.begin(), framebuffer.end(), 0xFFFFFFFF);
            std::fill(depthBuffer.begin(), depthBuffer.end(), FLT_MAX);
            
            renderFrame(camera, framebuffer, depthBuffer, spheres);
            drawSDL(renderer, texture, framebuffer);

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