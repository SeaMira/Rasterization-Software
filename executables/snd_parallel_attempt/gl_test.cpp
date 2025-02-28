#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glad/glad.h>
#include <SDL3/SDL.h>
#include <vector>



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

const int sphere_count = 512;

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
    // 64b
}; 

int main(int argc, char* argv[]) 
{
    Window window { title, SCR_WIDTH, SCR_HEIGHT, shown };
    
    ComputeShader glTestingShader("shaders/snd_parallel_attempt/gl_testing.compute");
    std::vector<SphereBillboard> sphB;
    for (int i = 0; i < sphere_count; i++)
    {
        sphB.push_back({glm::vec4((float)i, 0.0f, 0.0f, (float)i),
            glm::vec4(0.0f, 0.0f, 0.0f, (float)i),
            glm::vec4(0.0f, 0.0f, 0.0f, (float)i),
            glm::vec4(0.0f, 0.0f, 0.0f, (float)i),
            glm::vec4(-2.5f, 0.0f, 0.0f, (float)i)
        });
    }
    StorageBuffer sphereBillboardBuffer(GL_SHADER_STORAGE_BUFFER);
    sphereBillboardBuffer.generateBufferData(sphere_count*sizeof(SphereBillboard), 0, sphB.data(), GL_DYNAMIC_COPY);
    sphereBillboardBuffer.unbind();
    
    // testing shader
    glTestingShader.use();
    glTestingShader.setInt("sphereCount", sphere_count);
    glDispatchCompute((SCR_WIDTH + workGroupSizeX - 1) / 16, (SCR_HEIGHT + workGroupSizeY - 1) / 16, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);


    sphereBillboardBuffer.bind();
    SphereBillboard* mappedData = (SphereBillboard*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, sphere_count * sizeof(SphereBillboard), GL_MAP_READ_BIT);

    if (mappedData) {
        std::vector<SphereBillboard> sphB_CPU(mappedData, mappedData + sphere_count);  // Copiar datos al vector

        // Imprimir la componente .w de cameraSpaceSphPosR para verificar
        for (int i = 0; i < sphere_count; i++) {  // Solo imprimir los primeros 10 elementos
            std::cout << "Sphere " << i << " cameraSpaceSphPosR.w: " << sphB_CPU[i].cameraSpaceSphPosR.w 
            << " and downLeftCornerMinY.w " << sphB_CPU[i].downLeftCornerMinY.w << std::endl;
        }

        // Desmapear el buffer
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    } else {
        std::cerr << "Error mapping SSBO." << std::endl;
    }

    sphereBillboardBuffer.unbind();


    return 0;
}