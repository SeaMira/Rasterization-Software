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
int SCR_WIDTH = 1024;
int SCR_HEIGHT = 1024;
int sphere_count = 1024* 512;

std::string title = "Second Parallel Version"; 

bool shown = true;

GLuint workGroupSizeX = 16;  // Deifining threads-per-group (X)
GLuint workGroupSizeY = 16;  // Deifining threads-per-group (Y)


int visibleSpheresCount = 0;
int notOccludedSpheresCount = 0;

// downsample settings
int downsampleLevel = 4;
int downsampleWorkGroupSizeX = 16/downsampleLevel;
int downsampleWorkGroupSizeY = 16/downsampleLevel;

struct SphereBillboard 
{
    glm::vec4 cameraSpaceSphPosR;   // 16b - camera space position of the sphere and radius
    glm::vec4 upRightCornerMaxX;     // 16b - up right corner of the billboard in camera space and the max X coordinate in screen space 
    glm::vec4 upLeftCornerMaxY;      // 16b - up left corner of the billboard in camera space and the max Y coordinate in screen space
    glm::vec4 downRightCornerMinX;   // 16b - down right corner of the billboard in camera space and the min X coordinate in screen space
    glm::vec4 downLeftCornerMinY;    // 16b - down left corner of the billboard in camera space and the min Y coordinate in screen space
    uint index;
    int padding1, padding2, padding3;
    // 96b
}; 

int main(int argc, char* argv[]) 
{
    std::unordered_map<std::string, int*> scene_data = {
        {"Screen width", &SCR_WIDTH},
        {"Screen height", &SCR_HEIGHT},
        {"Sphere count", &sphere_count},
        {"Spheres On Frustum", &visibleSpheresCount},
        {"Drawn spheres", &notOccludedSpheresCount}
    };

    AppOpenGL window { title, SCR_WIDTH, SCR_HEIGHT, shown };
    Camera camera(SCR_WIDTH, SCR_HEIGHT);
    camera.SetPosition(.0f, .0f, .0f);
    CameraController camera_controller(window, camera);
    
    ComputeShader cleaningComputeShader("assets/shaders/snd_parallel_attempt/set_to_black.compute");
    #if CPU_FRUSTUM_CULLING
        ComputeShader bboxExtractionShader("assets/shaders/snd_parallel_attempt/bbox_extraction.compute");
        ComputeShader bboxIntersectionShader("assets/shaders/snd_parallel_attempt/bbox_intersect.compute");
    #else
        ComputeShader bboxExtractionShader("assets/shaders/snd_parallel_attempt/bbox_extraction_cull.compute");
        ComputeShader bboxIntersectionShader("assets/shaders/snd_parallel_attempt/bbox_intersect_cull.compute");
    #endif

    ComputeShader hizPyramidComputeShader("assets/shaders/snd_parallel_attempt/mipmap_gen.compute");
    ComputeShader pixelCountComputeShader("assets/shaders/snd_parallel_attempt/pixel_count.compute");

    Canvas canvas(GL_TEXTURE_2D, GL_RGBA8, SCR_WIDTH, SCR_HEIGHT);
    canvas.setFBO(GL_COLOR_ATTACHMENT0);
    
    if (!canvas.getFramebuffer().isComplete()) 
    {
        throw std::runtime_error("Error: Incomplete Framebuffer.");
    }
        
    std::vector<glm::vec4> spheres = getScene(sphere_count);
    sphere_count = spheres.size(); 
    std::vector<std::pair<glm::vec3, glm::vec3>> chkPoints = getCheckpoints(sphere_count, spheres);
    
    std::vector<SphereContainer> visibleSpheres(sphere_count);
    fillSpheresData(spheres, visibleSpheres);
    
    StorageBuffer sphereBuffer(GL_SHADER_STORAGE_BUFFER);
    sphereBuffer.generateBufferData(sphere_count * sizeof(SphereContainer), 1, 
    visibleSpheres.data(), GL_STATIC_DRAW);
    sphereBuffer.unbind();
    
    std::vector<SphereBillboard> billboards(sphere_count);
    StorageBuffer sphereBillboardBuffer(GL_SHADER_STORAGE_BUFFER);
    sphereBillboardBuffer.generateBufferData(sphere_count * sizeof(SphereBillboard), 2, 
    billboards.data(), GL_STATIC_DRAW);
    sphereBillboardBuffer.unbind();
    
    #if !CPU_FRUSTUM_CULLING
        GLuint zero = 0;
        GLuint resetValue = 0;
        StorageBuffer visibleSpheresAtomicCounter(GL_SHADER_STORAGE_BUFFER);
        visibleSpheresAtomicCounter.generateBufferData(sizeof(GLuint), 3, &zero, GL_STATIC_DRAW);
        visibleSpheresAtomicCounter.unbind();
    #endif

    Texture depthTexture(GL_TEXTURE_2D, GL_R32F, SCR_WIDTH, SCR_HEIGHT, 4);
    Texture downsampledDepthTexture(GL_TEXTURE_2D, GL_R32F, SCR_WIDTH/(1 << downsampleLevel), SCR_HEIGHT/(1 << downsampleLevel), 5);
    
    std::vector<GLuint> visibilityFrames(2 * sphere_count, 10);
    StorageBuffer visibilityFramesBuffer(GL_SHADER_STORAGE_BUFFER);
    visibilityFramesBuffer.generateBufferData(2 * sphere_count * sizeof(GLuint), 6, 
        visibilityFrames.data(), GL_STATIC_DRAW);
    visibilityFramesBuffer.unbind();
    
    std::vector<GLuint> pixelCountFrames(SCR_WIDTH*SCR_HEIGHT, 0);
    StorageBuffer pixelCountFramesBuffer(GL_SHADER_STORAGE_BUFFER);
    pixelCountFramesBuffer.generateBufferData(SCR_WIDTH*SCR_HEIGHT * sizeof(int), 7, 
        pixelCountFrames.data(), GL_STATIC_DRAW);
    pixelCountFramesBuffer.unbind();

    Benchmark benchmark(camera_controller, chkPoints);
    Profiler profiler(window, "media/off/fst_parallel/frame_times.off", "media/off/fst_parallel/process_times.off");
    
    window.setupSceneInfoGui("Scene Info", scene_data);
    window.setupCameraGui("Camera Info", &camera);
    window.setupInputInfoGui("General Input Info");
    window.setupBenchmarkInfoGui("Benchmark", &benchmark);

    // Calculating number of work groups (based on the number of threads and spheres)
    GLuint numGroupsX = (sphere_count + workGroupSizeX - 1) / workGroupSizeX;
    GLuint numGroupsY = 1;

    canvas.bindTexture();
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

            // hiz shader
            hizPyramidComputeShader.use();
            downsampledDepthTexture.bindImage(GL_READ_WRITE, GL_R32F);
            depthTexture.unbindImage(GL_READ_WRITE, GL_R32F);
            depthTexture.bind();
            glm::vec2 utexelDimensions = glm::vec2( 1.0f / (float)SCR_WIDTH, 1.0f / (float)SCR_HEIGHT );
            hizPyramidComputeShader.setInt("depthTextureSampler", 2);
            hizPyramidComputeShader.setVec2("utexelDimensions", utexelDimensions);
            glDispatchCompute((SCR_WIDTH + downsampleWorkGroupSizeX - 1) / downsampleWorkGroupSizeX, (SCR_HEIGHT + downsampleWorkGroupSizeY - 1) / downsampleWorkGroupSizeY, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

            depthTexture.bindImage(GL_READ_WRITE, GL_R32F);

            // cleaning shader
            cleaningComputeShader.use();
            cleaningComputeShader.setFloat("far", camera.getFar());
            cleaningComputeShader.setVec2I("screenResolution", screenResolution);
            glDispatchCompute((SCR_WIDTH + workGroupSizeX - 1) / 16, (SCR_HEIGHT + workGroupSizeY - 1) / 16, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

            downsampledDepthTexture.unbindImage(GL_READ_WRITE, GL_R32F);
            downsampledDepthTexture.bind();

            // bbox extraction shader
            bboxExtractionShader.use();
            Frustum frustum(camera);
            #if CPU_FRUSTUM_CULLING
                cullSpheres(spheres, visibleSpheres, frustum, visibleSpheresCount);
                sphereBuffer.bind();
                glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, visibleSpheresCount * sizeof(SphereContainer), visibleSpheres.data());
                sphereBuffer.unbind();
                numGroupsX = (visibleSpheresCount + workGroupSizeX - 1) / workGroupSizeX;
                
                bboxExtractionShader.setInt("sphereCount", visibleSpheresCount); // visible sphere count given
            #else
                visibleSpheresAtomicCounter.bind();
                void* mappedAtomic = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), GL_MAP_READ_BIT);
                if (mappedAtomic != nullptr) 
                {
                    GLuint* atomicValue = reinterpret_cast<GLuint*>(mappedAtomic);
                    notOccludedSpheresCount = *atomicValue;
                    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
                }
                glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
                visibleSpheresAtomicCounter.unbind();
                
                bboxExtractionShader.setInt("sphereCount", sphere_count);
                bboxExtractionShader.setVec4("frustumTopFace", glm::vec4(frustum.topFace.normal, frustum.topFace.distance));
                bboxExtractionShader.setVec4("frustumBottomFace", glm::vec4(frustum.bottomFace.normal, frustum.bottomFace.distance));
                bboxExtractionShader.setVec4("frustumRightFace", glm::vec4(frustum.rightFace.normal, frustum.rightFace.distance));
                bboxExtractionShader.setVec4("frustumLeftFace", glm::vec4(frustum.leftFace.normal, frustum.leftFace.distance));
                bboxExtractionShader.setVec4("frustumFarFace", glm::vec4(frustum.farFace.normal, frustum.farFace.distance));
                bboxExtractionShader.setVec4("frustumNearFace", glm::vec4(frustum.nearFace.normal, frustum.nearFace.distance));
            #endif

            bboxExtractionShader.setVec2I("screenResolution", screenResolution);
            bboxExtractionShader.setMat4("proj", camera.getProjection());
            bboxExtractionShader.setMat4("view", camera.getView());
            bboxExtractionShader.setVec3("up", camera.getUp());
            bboxExtractionShader.setVec3("front", camera.getFront());
            bboxExtractionShader.setVec3("cameraPos", camera.getPosition());
            glDispatchCompute(numGroupsX, numGroupsY, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            
            // bbox intersection shader
            bboxIntersectionShader.use();
            #if CPU_FRUSTUM_CULLING
                bboxIntersectionShader.setInt("sphereCount", visibleSpheresCount); // visible sphere count given
            #endif
            bboxIntersectionShader.setFloat("far", camera.getFar());
            bboxIntersectionShader.setVec2I("screenResolution", screenResolution);
            bboxIntersectionShader.setMat4("proj", camera.getProjection());
            bboxIntersectionShader.setVec3("cameraPos", camera.getPosition());
            glDispatchCompute((SCR_WIDTH + workGroupSizeX - 1) / 16, (SCR_HEIGHT + workGroupSizeY - 1) / 16, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

            pixelCountComputeShader.use();
            pixelCountComputeShader.setVec2I("screenResolution", screenResolution);
            glDispatchCompute((SCR_WIDTH + workGroupSizeX - 1) / 16, (SCR_HEIGHT + workGroupSizeY - 1) / 16, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            
            glBindFramebuffer(GL_READ_FRAMEBUFFER, canvas.getFramebuffer().getId());
            isRunning = window.update();

            if (window.getInput().isKeyDown(Key::F10)) 
            {
                std::string sshot_name  = "media/img/scnd_parallel/frame_" + std::to_string(benchmark.getCheckpointID()) + ".bmp";
                canvas.takeScreenshot(sshot_name);
            }

            ////// checking drawn spheres ///////
            
            // sphereBuffer.bind();  // Primero aseguramos que el buffer está vinculado

            // // mapping buffer on reading mode
            // void* mappedData = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 
            //                                     0, 
            //                                     spheres.size() * sizeof(SphereContainer), 
            //                                     GL_MAP_READ_BIT);

            // // verifying correct mapping
            // if (mappedData != nullptr) {
            //     // Accessing mapped buffer
            //     SphereContainer* spheresData = reinterpret_cast<SphereContainer*>(mappedData);
            //     visibleSpheresCount = 0;
            //     // looping on elements checking if drawn or not
            //     for (size_t i = 0; i < sphere_count; ++i)
            //         if (spheresData[i].wasDrawn[1]) 
            //             visibleSpheresCount++;
                
            //     // unmapping buffer once finished
            //     glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
            // } else {
            //     std::cerr << "Failed to map the buffer!" << std::endl;
            // }

            // sphereBuffer.unbind();
            
            ////// END:: checking drawn spheres ///////
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }


    return 0;
}