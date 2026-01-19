#include "core/rendering/gpu_cull_renderer.h"
#include "core/scene/scene.h"
#include "algorithms/frustum_cull.h"
#include "ux/camera.h"
#include "utils/parallel/aux_functions.h"

GPUCullRenderer::GPUCullRenderer(bool withOcclusion)
    : BaseComputeRenderer(withOcclusion ? "GPU Cull Renderer (with Occlusion)" : "GPU Cull Renderer")
    , m_withOcclusion(withOcclusion)
{
}

void GPUCullRenderer::onInitialize(const RenderConfig& config)
{
    // Setup canvas depth data
    if (m_withOcclusion)
    {
        m_canvas->setupDepthDownsample(
            config.getDownsampleLevel(),
            config.getScreenWidth(),
            config.getScreenHeight()
        );
    }
}

void GPUCullRenderer::createShaders()
{
    // Create cleaning shader
    m_cleaningShader = std::make_unique<ComputeShader>(
        getShaderPath("set_to_black").c_str(),
        "Cleaning Shader"
    );
    m_canvas->setupCleaningProgram(*m_cleaningShader);

    // Create sphere culling shader
    m_sphereShader = std::make_unique<ComputeShader>(
        getShaderPath("sphere_culling").c_str(),
        "Spheres Shader"
    );

    // Create cylinder culling shader
    m_cylinderShader = std::make_unique<ComputeShader>(
        getShaderPath("cylinder_culling").c_str(),
        "Cylinders Shader"
    );

    if (m_withOcclusion)
    {
        // Create Hi-Z pyramid shader
        m_hizPyramidShader = std::make_unique<ComputeShader>(
            getShaderPath("mipmap_gen").c_str(),
            "Hiz Pyramid Shader"
        );
        m_canvas->setupDownsamplingProgram(*m_hizPyramidShader);

        // Create pixel count shader
        m_pixelCountShader = std::make_unique<ComputeShader>(
            getShaderPath("pixel_count").c_str(),
            "Pixel Count Shader"
        );
    }
}

void GPUCullRenderer::createBuffers(Scene& scene)
{
    m_sphereCount = static_cast<int>(scene.getSphereCount());
    m_cylinderCount = static_cast<int>(scene.getCylinderCount());

    // Calculate work groups
    m_numGroupsSpheres = calculateWorkGroups(m_sphereCount, m_config.getWorkGroupSizeXPerSphere());
    m_numGroupsCylinders = calculateWorkGroups(m_cylinderCount, m_config.getWorkGroupSizeXPerCylinder());

    std::cout << "Creating buffers for " 
              << m_sphereCount << " spheres and " 
              << m_cylinderCount << " cylinders.\n";
    // Create sphere buffer (binding point 6)
    m_sphereBuffer = std::make_unique<StorageBuffer>(
        GL_SHADER_STORAGE_BUFFER,
        static_cast<int>(scene.getSphereDataSize()),
        6,
        scene.getSphereData(),
        GL_STATIC_DRAW
    );

    // Create cylinder buffer (binding point 7)
    m_cylinderBuffer = std::make_unique<StorageBuffer>(
        GL_SHADER_STORAGE_BUFFER,
        static_cast<int>(scene.getCylinderDataSize()),
        7,
        scene.getCylinderData(),
        GL_STATIC_DRAW
    );

    // Create atomic counters
    GLuint zero = 0;
    m_frustumAtomicCounter = std::make_unique<StorageBuffer>(
        GL_SHADER_STORAGE_BUFFER,
        sizeof(GLuint),
        8,
        &zero,
        GL_DYNAMIC_COPY
    );

    
    if (m_withOcclusion)
    {
        m_occlusionAtomicCounter = std::make_unique<StorageBuffer>(
            GL_SHADER_STORAGE_BUFFER,
            sizeof(GLuint),
            9,
            &zero,
            GL_DYNAMIC_COPY
        );
        // Create visibility frames buffer
        int totalEntities = m_sphereCount + m_cylinderCount;
        std::vector<GLuint> visibilityFrames(2 * totalEntities, 10);
        m_visibilityBuffer = std::make_unique<StorageBuffer>(
            GL_SHADER_STORAGE_BUFFER,
            static_cast<int>(2 * totalEntities * sizeof(GLuint)),
            5,
            visibilityFrames.data(),
            GL_DYNAMIC_COPY
        );

        // Create pixel count buffer
        m_pixelCountBuffer = std::make_unique<StorageBuffer>(
            GL_SHADER_STORAGE_BUFFER,
            m_config.getScreenWidth() * m_config.getScreenHeight() * sizeof(GLuint),
            4,
            nullptr,
            GL_DYNAMIC_COPY
        );
    }

    // Initialize result counts
    m_lastResult.visibleSphereCount = m_sphereCount;
    m_lastResult.drawnSphereCount = m_sphereCount;
    m_lastResult.visibleCylinderCount = m_cylinderCount;
    m_lastResult.drawnCylinderCount = m_cylinderCount;
}

void GPUCullRenderer::preRender(Camera& camera)
{
    // Downsample depth for Hi-Z if using occlusion
    if (m_withOcclusion)
    {
        m_canvas->downsampleCanvasDepth(
            m_config.getDownsampleWorkGroupSizeX(),
            m_config.getDownsampleWorkGroupSizeY()
        );
    }

    // Clean canvas buffers
    m_canvas->cleanCanvasBuffers(
        m_config.getWorkGroupSizeXPerPixel(),
        m_config.getWorkGroupSizeYPerPixel(),
        camera.getFar()
    );

    // Reset atomic counters
    resetAtomicCounters();
}

void GPUCullRenderer::renderSpheres(Scene& scene, Camera& camera, const Frustum& frustum)
{
    m_sphereShader->use();

#if BENCHMARKING
    readVisibilityCounts();
    m_sphereShader->setInt("benchmark", 1);
#else
    m_sphereShader->setInt("benchmark", 0);
#endif

    m_sphereShader->setInt("sphereCount", m_sphereCount);
    m_sphereShader->setVec2I("screenResolution", m_screenResolution);
    
    if (m_withOcclusion)
    {
        m_sphereShader->setUint("visibilityFrameBufferIndexOffset", 0);
    }

    setFrustumUniformsOnShader(*m_sphereShader, frustum);
    setCameraUniformsOnShader(*m_sphereShader, camera);

    dispatchCompute(
        m_numGroupsSpheres, 1,
        "Sphere Shader",
        GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT
    );
}

void GPUCullRenderer::renderCylinders(Scene& scene, Camera& camera, const Frustum& frustum)
{
    m_cylinderShader->use();

#if BENCHMARKING
    // Read sphere counts before cylinder pass
    m_frustumAtomicCounter->bind();
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &m_lastResult.visibleSphereCount);
    GLuint resetValue = 0;
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
    m_frustumAtomicCounter->unbind();

    if (m_withOcclusion)
    {
        m_occlusionAtomicCounter->bind();
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &m_lastResult.drawnSphereCount);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &resetValue);
        m_occlusionAtomicCounter->unbind();
    }
    else
    {
        m_lastResult.drawnSphereCount = m_lastResult.visibleSphereCount;
    }

    m_cylinderShader->setInt("benchmark", 1);
#else
    m_cylinderShader->setInt("benchmark", 0);
#endif

    m_cylinderShader->setInt("cylinderCount", m_cylinderCount);
    m_cylinderShader->setVec2I("screenResolution", m_screenResolution);

    if (m_withOcclusion)
    {
        m_cylinderShader->setUint("visibilityFrameBufferIndexOffset", static_cast<GLuint>(m_sphereCount));
    }

    setFrustumUniformsOnShader(*m_cylinderShader, frustum);
    setCameraUniformsOnShader(*m_cylinderShader, camera);

    dispatchCompute(
        m_numGroupsCylinders, 1,
        "Cylinder Shader",
        GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT
    );
}

void GPUCullRenderer::postRender()
{
    if (m_withOcclusion && m_pixelCountShader)
    {
        m_pixelCountShader->use();
        m_pixelCountShader->setVec2I("screenResolution", m_screenResolution);

        GLuint groupsX = calculateWorkGroups(m_config.getScreenWidth(), m_config.getWorkGroupSizeXPerPixel());
        GLuint groupsY = calculateWorkGroups(m_config.getScreenHeight(), m_config.getWorkGroupSizeYPerPixel());

        dispatchCompute(
            groupsX, groupsY,
            "Pixel Count Shader",
            GL_SHADER_IMAGE_ACCESS_BARRIER_BIT
        );
    }

#if BENCHMARKING
    // Read final cylinder counts
    m_frustumAtomicCounter->bind();
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &m_lastResult.visibleCylinderCount);
    m_frustumAtomicCounter->unbind();

    if (m_withOcclusion)
    {
        m_occlusionAtomicCounter->bind();
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &m_lastResult.drawnCylinderCount);
        m_occlusionAtomicCounter->unbind();
    }
    else
    {
        m_lastResult.drawnCylinderCount = m_lastResult.visibleCylinderCount;
    }
#endif
}

void GPUCullRenderer::onCleanup()
{
    m_sphereShader.reset();
    m_cylinderShader.reset();
    m_cleaningShader.reset();
    m_hizPyramidShader.reset();
    m_pixelCountShader.reset();

    m_sphereBuffer.reset();
    m_cylinderBuffer.reset();
    m_visibilityBuffer.reset();
    m_frustumAtomicCounter.reset();
    m_occlusionAtomicCounter.reset();
    m_pixelCountBuffer.reset();
}

std::string GPUCullRenderer::getShaderPath(const std::string& shaderName) const
{
    std::string basePath = "assets/shaders/fst_parallel_attempt/gpu_cull/";
    std::string suffix = m_withOcclusion ? ".compute" : "_no_occ.compute";
    return basePath + shaderName + suffix;
}

void GPUCullRenderer::resetAtomicCounters()
{
    GLuint zero = 0;
    
    m_frustumAtomicCounter->bind();
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &zero);
    m_frustumAtomicCounter->unbind();

    if (m_withOcclusion)
    {
        m_occlusionAtomicCounter->bind();
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &zero);
        m_occlusionAtomicCounter->unbind();
    }
}

void GPUCullRenderer::readVisibilityCounts()
{
    // This is called during benchmarking to get previous frame's counts
}
