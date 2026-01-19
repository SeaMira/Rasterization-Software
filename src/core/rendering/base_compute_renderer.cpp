#include "core/rendering/base_compute_renderer.h"
#include "core/scene/scene.h"
#include "core/interfaces/i_culling_strategy.h"
#include "algorithms/frustum_cull.h"
#include "ux/camera.h"

BaseComputeRenderer::BaseComputeRenderer(const std::string& name)
    : m_name(name)
{
}

void BaseComputeRenderer::initialize(const RenderConfig& config)
{
    m_config = config;
    m_screenResolution = glm::ivec2(config.getScreenWidth(), config.getScreenHeight());
    
    // Create the canvas
    m_canvas = std::make_unique<Canvas>(
        GL_TEXTURE_2D, GL_RGBA8, 
        config.getScreenWidth(), config.getScreenHeight()
    );
    m_canvas->setFBO(GL_COLOR_ATTACHMENT0);
    m_canvas->setupDepthData(config.getScreenWidth(), config.getScreenHeight());

    // Call subclass initialization
    onInitialize(config);
    
    // Create shaders
    createShaders();

    m_canvas->bindFBO();
    m_canvas->bindTexture();
    
    // Initialize culling strategy if present
    if (m_cullingStrategy)
    {
        m_cullingStrategy->initialize(config.getScreenWidth(), config.getScreenHeight());
    }
    
    m_initialized = true;
}

CullingResult BaseComputeRenderer::render(Scene& scene, Camera& camera)
{
    if (!m_initialized)
    {
        return CullingResult{};
    }

    // Bind canvas
    // m_canvas->bindTexture();
    // m_canvas->bindFBO();
    
    // Create frustum from camera
    Frustum frustum(camera);

    // Pre-render phase
    preRender(camera);

    // Render spheres
    renderSpheres(scene, camera, frustum);

    // Render cylinders
    renderCylinders(scene, camera, frustum);

    // Post-render phase
    postRender();

    // Bind framebuffer for reading
    glBindFramebuffer(GL_READ_FRAMEBUFFER, m_canvas->getFramebuffer().getId());

    return m_lastResult;
}

void BaseComputeRenderer::cleanup()
{
    onCleanup();
    
    if (m_cullingStrategy)
    {
        m_cullingStrategy->cleanup();
    }
    
    m_canvas.reset();
    m_initialized = false;
    m_buffersCreated = false;
}

void BaseComputeRenderer::setCullingStrategy(std::unique_ptr<ICullingStrategy> strategy)
{
    m_cullingStrategy = std::move(strategy);
    
    if (m_initialized && m_cullingStrategy)
    {
        m_cullingStrategy->initialize(
            m_config.getScreenWidth(), 
            m_config.getScreenHeight()
        );
    }
}

void BaseComputeRenderer::setFrustumUniformsOnShader(ComputeShader& shader, const Frustum& frustum)
{
    // Set frustum plane uniforms with the names expected by the shaders
    shader.setVec4("frustumTopFace", glm::vec4(frustum.topFace.normal, frustum.topFace.distance));
    shader.setVec4("frustumBottomFace", glm::vec4(frustum.bottomFace.normal, frustum.bottomFace.distance));
    shader.setVec4("frustumRightFace", glm::vec4(frustum.rightFace.normal, frustum.rightFace.distance));
    shader.setVec4("frustumLeftFace", glm::vec4(frustum.leftFace.normal, frustum.leftFace.distance));
    shader.setVec4("frustumFarFace", glm::vec4(frustum.farFace.normal, frustum.farFace.distance));
    shader.setVec4("frustumNearFace", glm::vec4(frustum.nearFace.normal, frustum.nearFace.distance));
}

void BaseComputeRenderer::setCameraUniformsOnShader(ComputeShader& shader, Camera& camera)
{
    shader.setMat4("view", camera.getView());
    shader.setMat4("proj", camera.getProjection());
    shader.setVec3("cameraPos", camera.getPosition());
    shader.setVec3("front", camera.getFront());
    shader.setVec3("up", camera.getUp());
    shader.setVec3("right", camera.getRight());
    shader.setFloat("near", camera.getNear());
    shader.setFloat("far", camera.getFar());
    shader.setFloat("fov", camera.getFov());
}

void BaseComputeRenderer::dispatchCompute(GLuint numGroupsX, GLuint numGroupsY, 
                                          const std::string& label, GLbitfield barriers)
{
    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, label.c_str());
    glDispatchCompute(numGroupsX, numGroupsY, 1);
    glMemoryBarrier(barriers);
    glPopDebugGroup();
}

GLuint BaseComputeRenderer::calculateWorkGroups(int count, GLuint groupSize) const
{
    return static_cast<GLuint>((count + groupSize - 1) / groupSize);
}
