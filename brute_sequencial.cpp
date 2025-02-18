#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glad/glad.h>
#include <SDL3/SDL.h>
#include <vector>
#include <cmath>

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

const int sphere_count = 10;

std::string title = "Brute sequencial method"; 

bool shown = true;

struct ProjectionResult
{
    float area;      
    glm::vec2  center;    
    glm::vec2  axisA;     
    glm::vec2  axisB;
    // implicit ellipse f(x,y) = a·x² + b·x·y + c·y² + d·x + e·y + f = 0 */
    float a, b, c, d, e, f; 
};


float iSphere(glm::vec3 ro, glm::vec3 rd, glm::vec3 sph, float radius )
{
    float a = glm::dot(rd, rd);
    float b = glm::dot( rd, sph );
    float c = glm::dot( sph, sph ) - radius*radius;
    float h = b*b - a * c;
    if( h < 0.0f ) return -1.0f;
    return (b - std::sqrt( h ))/a;
}


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

ProjectionResult projectSphere( /* sphere */ glm::vec4 sph, 
    /* camera matrix */ glm::mat4 cam,
    /* projection    */ float fle )
{
// transform to camera space	
glm::vec3  sph3(sph[0], sph[1], sph[2]);
glm::vec3  o = glm::vec3(cam*glm::vec4(sph3, 1.0f));

float r2 = sph[3]*sph[3];
float z2 = o[2]*o[2];	
float l2 = dot(o,o);

float r2z2 = r2-z2;
float area = -3.141593*fle*fle*r2*sqrt(abs((l2-r2)/(r2z2)))/(r2z2 + 1e-12);

// axis
glm::vec2 axa = fle*sqrt(-r2*(r2-l2)/((l2-z2)*(r2z2)*(r2z2) + 1e-12f)) * glm::vec2( o[0],o[1]);
glm::vec2 axb = fle*sqrt(-r2/((l2-z2)*(r2z2) + 1e-6f))*glm::vec2(-o[1],o[0]);

// center
glm::vec2  cen = fle*o.z*glm::vec2(o[0], o[1])/(z2-r2);


return { area, cen, axa, axb, 
/* a */ r2 - o.y*o.y - z2,
/* b */ 2.0f*o.x*o.y,
/* c */ r2 - o.x*o.x - z2,
/* d */ -2.0f*o.x*o.z*fle,
/* e */ -2.0f*o.y*o.z*fle,
/* f */ (r2-l2+z2)*fle*fle };

}

uint32_t depthToColor(float depth, float minDepth, float maxDepth) 
{
    // if (maxDepth == minDepth) return 0xFFFFFF; // Evitar divisiones por cero

    // // Normalizar la profundidad a un rango de 0 a 1 usando la distancia real
    // float normalizedDepth = (depth - minDepth) / (maxDepth - minDepth);
    // normalizedDepth = glm::clamp(normalizedDepth, 0.0f, 1.0f);

    // // Convertir a un gradiente de colores (por ejemplo, de azul a rojo)
    // uint8_t red = static_cast<uint8_t>(normalizedDepth * 255.0f);
    // uint8_t blue = static_cast<uint8_t>((1.0f - normalizedDepth) * 255.0f);
    uint8_t color = static_cast<uint8_t>(255.0f/(depth + 1.0f));

    // std::cout << depth << std::endl;

    return (0xFF000000) | (color << 16) | (color << 8) | (color); // Gradiente de rojo a azul
}

void renderFrame(std::vector<uint32_t>& framebuffer, 
    std::vector<float>& depthBuffer, std::vector<std::pair<glm::vec3, float>>& spheres,
    Camera& cam) 
{
    float minDepth = 0.0f, maxDepth = 10.0f;

    glm::mat4& proj = cam.getProjection();
    glm::mat4& view = cam.getView();
    glm::vec3& up = cam.getUp();
    glm::vec3& right = cam.getRight();
    glm::vec3& front = cam.getFront();
    glm::vec3& camPos = cam.getPosition();
    glm::vec2 resol = glm::vec2(SCR_WIDTH, SCR_HEIGHT);
    float fov = cam.getFov();
    
    for (const auto& sphere : spheres) 
    {
        ProjectionResult sphRes = projectSphere(glm::vec4(sphere.first, sphere.second), view, glm::radians(fov));
        sphRes.center = (-sphRes.center + glm::vec2(1.0f, 1.0f)) * resol /2.0f;
        sphRes.center[0] = std::floor(sphRes.center[0]);
        sphRes.center[1] = std::floor(sphRes.center[1]);
        
        sphRes.axisA = (-sphRes.axisA + glm::vec2(1.0f, 1.0f)) * resol /2.0f;
        sphRes.axisA[0] = std::floor(sphRes.axisA[0]);
        sphRes.axisA[1] = std::floor(sphRes.axisA[1]);
        sphRes.axisB = (-sphRes.axisB + glm::vec2(1.0f, 1.0f)) * resol /2.0f;
        sphRes.axisB[0] = std::floor(sphRes.axisB[0]);
        sphRes.axisB[1] = std::floor(sphRes.axisB[1]);

        glm::vec3 cameraSpaceSphere = glm::vec3(view * glm::vec4(sphere.first, 1.0f));
        glm::vec3 normCamSpaceSphere = glm::normalize(cameraSpaceSphere);
        glm::vec3 camImposPos = cameraSpaceSphere - normCamSpaceSphere * sphere.second;

        float sinAngle = sphere.second / (glm::length(cameraSpaceSphere) + 1e-6f);
        float tanAngle = std::tan(std::asin(sinAngle));
        float quadScale = tanAngle * glm::length(camImposPos);

        glm::vec3 impU = glm::normalize(glm::cross(normCamSpaceSphere, up));
        glm::vec3 impV = glm::cross(impU, normCamSpaceSphere) * quadScale;
        impU *= quadScale;

        glm::vec3 upRightCorner = camImposPos + impU + impV;
        glm::vec3 upLeftCorner = camImposPos - impU + impV;
        glm::vec3 downRightCorner = camImposPos + impU - impV;
        glm::vec3 downLeftCorner = camImposPos - impU - impV;

        glm::vec4 upRight = proj * glm::vec4(upRightCorner, 1.0f);
        glm::vec4 upLeft = proj * glm::vec4(upLeftCorner, 1.0f);
        glm::vec4 downRight = proj * glm::vec4(downRightCorner, 1.0f);
        glm::vec4 downLeft = proj * glm::vec4(downLeftCorner, 1.0f);
        
        glm::vec3 ndcUpRight = glm::vec3(upRightCorner) / upRight.w;
        glm::vec3 ndcUpLeft = glm::vec3(upLeftCorner) / upLeft.w;
        glm::vec3 ndcDownRight = glm::vec3(downRightCorner) / downRight.w;
        glm::vec3 ndcDownLeft = glm::vec3(downLeftCorner) / downLeft.w;

        glm::vec2 minCorner = glm::vec2(std::min(std::min(ndcUpRight[0], ndcUpLeft[0]), std::min(ndcDownRight[0], ndcDownLeft[0])),
                                        std::min(std::min(ndcUpRight[1], ndcUpLeft[1]), std::min(ndcDownRight[1], ndcDownLeft[1])));
        glm::vec2 maxCorner = glm::vec2(std::max(std::max(ndcUpRight[0], ndcUpLeft[0]), std::max(ndcDownRight[0], ndcDownLeft[0])),
                                        std::max(std::max(ndcUpRight[1], ndcUpLeft[1]), std::max(ndcDownRight[1], ndcDownLeft[1])));

        glm::ivec2 screenMin, screenMax;
        screenMin.x = (minCorner.x * 0.5f + 0.5f) * SCR_WIDTH;
        screenMin.y = (minCorner.y * 0.5f + 0.5f) * SCR_HEIGHT;
        screenMax.x = (maxCorner.x * 0.5f + 0.5f) * SCR_WIDTH;
        screenMax.y = (maxCorner.y * 0.5f + 0.5f) * SCR_HEIGHT;

        // std::cout << "AxisB " << sphRes.axisB[0] << ", " << sphRes.axisB[1] << std::endl;
        // BoundingBox bbox = getEllipseBoundingBox(sphRes);
        if (sphRes.area >0.0f) 
        {
            float difx = screenMax.x - screenMin.x;
            float dify = screenMax.y - screenMin.y;
            // std::cout << "screenMin.x "<< screenMin.x << std::endl;
            // std::cout << "screenMin.y "<< screenMin.y << std::endl;
            // std::cout << "screenMax.x "<< screenMax.x << std::endl;
            // std::cout << "screenMax.y "<< screenMax.y << std::endl;
            for (int px = screenMin.x; px < screenMax.x; px++)
            {
                for (int py = screenMin.y; py < screenMax.y; py++)
                {
                    if (px < 0 || px >= SCR_WIDTH || py <= 0 || py > SCR_HEIGHT) 
                        continue; 

                    // glm::vec2 uv = glm::vec2((float)px, (float)py) * 2.0f /glm::vec2((float)SCR_WIDTH, (float)SCR_HEIGHT) - 1.0f;
                    // glm::vec3 rd = glm::normalize( uv.x * right + uv.y * up + front );

                    float u = (float)(px - screenMin.x)/ difx;
                    float v = (float)(py - screenMin.y)/ dify;
                    glm::vec3 A = glm::mix(upLeftCorner, upRightCorner, u);
                    glm::vec3 B = glm::mix(downLeftCorner, downRightCorner, u);
                    glm::vec3 viewImpPos = glm::mix(A, B, v);
                    float h = iSphere(camPos, viewImpPos, cameraSpaceSphere, sphere.second);

                    // bool isInside = isInsideEllipse(px, py, sphRes);
                    int index = (SCR_HEIGHT - py) * SCR_WIDTH + px;
                    bool showBbox = (px == screenMin.x || py == screenMin.y || px == screenMax.x - 1 || py == screenMax.y - 1);
                    if ((h > 0.0f && depthBuffer[index] > h) || showBbox) 
                    {
                        depthBuffer[index] = h;
                        framebuffer[index] = depthToColor(h, minDepth, maxDepth);
                    }
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
    camera.SetPosition(.0f, .0f, .0f);
    CameraController camera_controller(window, camera);
    
    std::vector<uint32_t> framebuffer(SCR_WIDTH * SCR_HEIGHT);
    std::vector<float> depthBuffer(SCR_WIDTH * SCR_HEIGHT, FLT_MAX);
    
    SDL_Renderer* renderer = SDL_CreateRenderer(window.getHandle(), nullptr);
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, SCR_WIDTH, SCR_HEIGHT);
   
    std::filesystem::path path = "molecules/1AGA.mmtf";
    ChemFilesLoader loader(path);
    std::vector<std::pair<glm::vec3, float>> positions = loader.getSphereInfo();
    // std::vector<std::pair<glm::vec3, float>> spheres(positions.begin(), positions.begin() + std::min(positions.size(), static_cast<size_t>(sphere_count)));
    std::vector<std::pair<glm::vec3, float>> spheres {
        {glm::vec3(0.0f, 0.0f, 1.0f),  0.3f},
        {glm::vec3(-2.0f, 0.0f, 1.0f), 0.3f},
        // {glm::vec3(-1.0f, 0.0f, 1.0f), 0.3f},
        // {glm::vec3(0.0f, 0.0f, -1.0f), 0.7f},
        // {glm::vec3(1.0f, 0.0f, -1.0f), 0.7f},
        // {glm::vec3(2.0f, 0.0f, -1.0f), 0.7f},
        // {glm::vec3(3.0f, 0.0f, -1.0f), 0.7f},
        // {glm::vec3(4.0f, 0.0f, -1.0f), 0.7f}
    };
    
    try
    {
        bool isRunning = true;
        while ( isRunning )
        {
            
            // if (show_fps) std::cout << 1/window.getInput().deltaTime << std::endl;
            
            camera_controller.keyBoardAction();
            camera_controller.mouseAction();

            // glm::vec3 camFront = camera.getFront();
            // std::cout << camFront.x << ", " << camFront.y << ", " << camFront.z << std::endl;

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