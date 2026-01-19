#ifndef _BASE_COMPUTE_RENDERER_H_
#define _BASE_COMPUTE_RENDERER_H_

#include <memory>
#include <string>
#include <glm/glm.hpp>

#include "core/interfaces/i_renderer.h"
#include "core/interfaces/i_culling_strategy.h"
#include "core/config/render_config.h"
#include "vis/canvas.h"
#include "vis/compute_shader_program.h"
#include "vis/gl/storage_buffer.h"

// Forward declarations
class Scene;
class Camera;
class Frustum;

/**
 * @class BaseComputeRenderer
 * 
 * @brief Base class for compute shader-based renderers.
 * 
 * Provides common functionality for compute shader rendering pipelines.
 * Follows the Template Method Pattern by defining the skeleton of the
 * rendering algorithm and allowing subclasses to override specific steps.
 * 
 * Also follows the Single Responsibility Principle (SRP) by handling
 * only the rendering pipeline orchestration.
 */
class BaseComputeRenderer : public IRenderer
{
public:
    /**
     * @brief Constructs the base renderer.
     * 
     * @param name Renderer name for identification.
     */
    explicit BaseComputeRenderer(const std::string& name);

    /**
     * @brief Virtual destructor.
     */
    ~BaseComputeRenderer() override = default;

    /**
     * @brief Initializes the renderer.
     * 
     * @param config Rendering configuration.
     */
    void initialize(const RenderConfig& config) override;

    /**
     * @brief Hook for creating GPU buffers.
     * 
     * @param scene Scene containing data to buffer.
     */
    virtual void createBuffers(Scene& scene) = 0;

    /**
     * @brief Main render method implementing the Template Method pattern.
     * 
     * @param scene Scene to render.
     * @param camera Camera for view/projection.
     * @return CullingResult with visibility statistics.
     */
    CullingResult render(Scene& scene, Camera& camera) override;

    /**
     * @brief Cleans up resources.
     */
    void cleanup() override;

    /**
     * @brief Gets the renderer name.
     * 
     * @return Renderer name string.
     */
    std::string getName() const override { return m_name; }

    /**
     * @brief Checks initialization status.
     * 
     * @return True if initialized.
     */
    bool isInitialized() const override { return m_initialized; }

    /**
     * @brief Sets the culling strategy.
     * 
     * @param strategy Culling strategy implementation.
     */
    void setCullingStrategy(std::unique_ptr<ICullingStrategy> strategy);

protected:
    // Template Method hooks - subclasses override these

    /**
     * @brief Hook for subclass-specific initialization.
     * 
     * @param config Rendering configuration.
     */
    virtual void onInitialize(const RenderConfig& config) = 0;

    /**
     * @brief Hook for creating required shaders.
     */
    virtual void createShaders() = 0;

    /**
     * @brief Hook for pre-render operations (e.g., canvas cleaning).
     * 
     * @param camera Camera for the current frame.
     */
    virtual void preRender(Camera& camera) = 0;

    /**
     * @brief Hook for rendering spheres.
     * 
     * @param scene Scene containing spheres.
     * @param camera Camera for view/projection.
     * @param frustum View frustum for culling.
     */
    virtual void renderSpheres(Scene& scene, Camera& camera, const Frustum& frustum) = 0;

    /**
     * @brief Hook for rendering cylinders.
     * 
     * @param scene Scene containing cylinders.
     * @param camera Camera for view/projection.
     * @param frustum View frustum for culling.
     */
    virtual void renderCylinders(Scene& scene, Camera& camera, const Frustum& frustum) = 0;

    /**
     * @brief Hook for post-render operations.
     */
    virtual void postRender() = 0;

    /**
     * @brief Hook for subclass-specific cleanup.
     */
    virtual void onCleanup() = 0;

    // Utility methods for subclasses

    /**
     * @brief Sets common frustum uniforms on a shader.
     * 
     * @param shader Shader to configure.
     * @param frustum Frustum data.
     */
    void setFrustumUniformsOnShader(ComputeShader& shader, const Frustum& frustum);

    /**
     * @brief Sets common camera uniforms on a shader.
     * 
     * @param shader Shader to configure.
     * @param camera Camera data.
     */
    void setCameraUniformsOnShader(ComputeShader& shader, Camera& camera);

    /**
     * @brief Dispatches a compute shader with debug labeling.
     * 
     * @param numGroupsX Number of work groups in X.
     * @param numGroupsY Number of work groups in Y.
     * @param label Debug label for the dispatch.
     * @param barriers Memory barrier flags.
     */
    void dispatchCompute(GLuint numGroupsX, GLuint numGroupsY, 
                         const std::string& label, GLbitfield barriers);

    /**
     * @brief Calculates the number of work groups needed.
     * 
     * @param count Number of elements to process.
     * @param groupSize Size of each work group.
     * @return Number of work groups.
     */
    GLuint calculateWorkGroups(int count, GLuint groupSize) const;

protected:
    std::string m_name;                             ///< Renderer name
    bool m_initialized = false;                     ///< Initialization flag
    bool m_buffersCreated = false;                  ///< Buffer creation flag
    
    RenderConfig m_config;                          ///< Render configuration
    std::unique_ptr<Canvas> m_canvas;               ///< Render canvas
    std::unique_ptr<ICullingStrategy> m_cullingStrategy; ///< Culling strategy
    
    CullingResult m_lastResult;                     ///< Last culling result
    glm::ivec2 m_screenResolution;                  ///< Screen resolution cache
};

#endif // _BASE_COMPUTE_RENDERER_H_
