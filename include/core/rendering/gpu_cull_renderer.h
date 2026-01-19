#ifndef _GPU_CULL_RENDERER_H_
#define _GPU_CULL_RENDERER_H_

#include "core/rendering/base_compute_renderer.h"
#include "vis/compute_shader_program.h"
#include "vis/gl/storage_buffer.h"

#include <memory>
#include <vector>

/**
 * @class GPUCullRenderer
 * 
 * @brief Compute shader renderer with GPU-based frustum culling.
 * 
 * Implements the BaseComputeRenderer with GPU frustum culling for
 * spheres and cylinders. This renderer performs culling on the GPU
 * using compute shaders, which is efficient for large numbers of
 * primitives.
 * 
 * Follows the Open/Closed Principle (OCP) - this class extends
 * BaseComputeRenderer without modifying it.
 */
class GPUCullRenderer : public BaseComputeRenderer
{
public:
    /**
     * @brief Constructs the GPU cull renderer.
     * 
     * @param withOcclusion Enable Hi-Z occlusion culling.
     */
    explicit GPUCullRenderer(bool withOcclusion = false);

    /**
     * @brief Destructor.
     */
    ~GPUCullRenderer() override = default;

protected:
    // BaseComputeRenderer hooks

    void onInitialize(const RenderConfig& config) override;
    void createShaders() override;
    void createBuffers(Scene& scene) override;
    void preRender(Camera& camera) override;
    void renderSpheres(Scene& scene, Camera& camera, const Frustum& frustum) override;
    void renderCylinders(Scene& scene, Camera& camera, const Frustum& frustum) override;
    void postRender() override;
    void onCleanup() override;

private:
    /**
     * @brief Gets the shader path based on occlusion setting.
     * 
     * @param shaderName Base shader name.
     * @return Full shader path.
     */
    std::string getShaderPath(const std::string& shaderName) const;

    /**
     * @brief Resets atomic counters for a new frame.
     */
    void resetAtomicCounters();

    /**
     * @brief Reads visibility counts from atomic counters.
     */
    void readVisibilityCounts();

private:
    bool m_withOcclusion;   ///< Occlusion culling enabled flag
    
    // Shaders
    std::unique_ptr<ComputeShader> m_sphereShader;
    std::unique_ptr<ComputeShader> m_cylinderShader;
    std::unique_ptr<ComputeShader> m_cleaningShader;
    std::unique_ptr<ComputeShader> m_hizPyramidShader;
    std::unique_ptr<ComputeShader> m_pixelCountShader;

    // Buffers
    std::unique_ptr<StorageBuffer> m_sphereBuffer;
    std::unique_ptr<StorageBuffer> m_cylinderBuffer;
    std::unique_ptr<StorageBuffer> m_visibilityBuffer;
    std::unique_ptr<StorageBuffer> m_frustumAtomicCounter;
    std::unique_ptr<StorageBuffer> m_occlusionAtomicCounter;
    std::unique_ptr<StorageBuffer> m_pixelCountBuffer;

    // Work group calculations
    GLuint m_numGroupsSpheres = 0;
    GLuint m_numGroupsCylinders = 0;

    // Entity counts
    int m_sphereCount = 0;
    int m_cylinderCount = 0;
};

#endif // _GPU_CULL_RENDERER_H_
