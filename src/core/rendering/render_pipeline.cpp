#include "core/rendering/render_pipeline.h"
#include "core/scene/scene.h"
#include "core/interfaces/i_culling_strategy.h"

RenderPipeline::RenderPipeline(const RenderConfig& config)
    : m_config(config)
{
    // Create window
    std::string title = config.getTitle();
    if (config.hasOcclusionCulling())
    {
        title += " - With Occlusion Culling";
    }
    else
    {
        title += " - Without Occlusion Culling";
    }
    
    m_window = std::make_unique<AppOpenGL>(
        title,
        config.getScreenWidth(),
        config.getScreenHeight(),
        config.isShown()
    );

    // Create camera
    m_camera = std::make_unique<Camera>(
        config.getScreenWidth(),
        config.getScreenHeight()
    );
    m_camera->SetPosition(0.0f, 0.0f, 0.0f);

    // Create camera controller
    m_cameraController = std::make_unique<CameraController>(*m_window, *m_camera);
}

RenderPipeline::~RenderPipeline()
{
    if (m_renderer)
    {
        m_renderer->cleanup();
    }
}

RenderPipeline::RenderPipeline(RenderPipeline&& other) noexcept = default;
RenderPipeline& RenderPipeline::operator=(RenderPipeline&& other) noexcept = default;

void RenderPipeline::initialize(std::unique_ptr<Scene> scene)
{
    m_scene = std::move(scene);

    // Update stats with scene info
    m_stats.visibleSphereCount = static_cast<int>(m_scene->getSphereCount());
    m_stats.drawnSphereCount = m_stats.visibleSphereCount;
    m_stats.visibleCylinderCount = static_cast<int>(m_scene->getCylinderCount());
    m_stats.drawnCylinderCount = m_stats.visibleCylinderCount;

    // Initialize renderer if set
    if (m_renderer)
    {
        std::cout << "Initializing renderer: " << m_renderer->getName() << "\n";
        m_renderer->initialize(m_config);
    }

    // Setup UI
    setupUI();

    m_initialized = true;
}

void RenderPipeline::setRenderer(std::unique_ptr<IRenderer> renderer)
{
    m_renderer = std::move(renderer);
    
    if (m_initialized && m_renderer)
    {
        m_renderer->initialize(m_config);
    }
}

void RenderPipeline::enableProfiling(const std::string& frameTimesFile,
                                     const std::string& processTimesFile,
                                     double timerDuration)
{
    int downsampleLevel = m_config.hasOcclusionCulling() ? m_config.getDownsampleLevel() : 0;
    
    m_profiler = std::make_unique<Profiler>(
        *m_window,
        frameTimesFile,
        processTimesFile,
        static_cast<int>(m_scene ? m_scene->getSphereCount() : 0),
        static_cast<int>(m_scene ? m_scene->getCylinderCount() : 0),
        downsampleLevel,
        timerDuration
    );
}

void RenderPipeline::enableBenchmark(const std::vector<std::pair<glm::vec3, glm::vec3>>& checkpoints)
{
    m_benchmark = std::make_unique<Benchmark>(*m_cameraController, checkpoints);
}

void RenderPipeline::setFrameCallback(FrameCallback callback)
{
    m_frameCallback = std::move(callback);
}

void RenderPipeline::run()
{
    if (!m_initialized)
    {
        throw std::runtime_error("RenderPipeline not initialized");
    }

    m_renderer->createBuffers(*m_scene);

    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

    while (executeFrame())
    {
        // Continue running
    }
}

bool RenderPipeline::executeFrame()
{
    // Update camera and benchmark
    updateCameraAndBenchmark();

    // Update profiler
    updateProfiler();

    // Handle input
    handleInput();

    // Render if we have a renderer
    if (m_renderer && m_scene)
    {
        CullingResult result = m_renderer->render(*m_scene, *m_camera);
        
        m_stats.visibleSphereCount = result.visibleSphereCount;
        m_stats.drawnSphereCount = result.drawnSphereCount;
        m_stats.visibleCylinderCount = result.visibleCylinderCount;
        m_stats.drawnCylinderCount = result.drawnCylinderCount;
    }

    // Update checkpoint
    if (m_benchmark)
    {
        m_stats.currentCheckpoint = m_benchmark->getCheckpointID();
    }

    // Update window and check if should continue
    bool running = m_window->update();

    // Call frame callback if set
    if (m_frameCallback)
    {
        running = running && m_frameCallback(m_stats);
    }

    return running;
}

void RenderPipeline::setupUI()
{
    // Create scene data map for UI
    static std::unordered_map<std::string, int*> sceneData;
    
    // Store as static to avoid dangling pointers
    static int screenWidth, screenHeight, sphereCount, spheresOnFrustum, 
               drawnSpheres, cylinderCount, cylindersOnFrustum, drawnCylinders;
    
    screenWidth = m_config.getScreenWidth();
    screenHeight = m_config.getScreenHeight();
    sphereCount = static_cast<int>(m_scene ? m_scene->getSphereCount() : 0);
    spheresOnFrustum = m_stats.visibleSphereCount;
    drawnSpheres = m_stats.drawnSphereCount;
    cylinderCount = static_cast<int>(m_scene ? m_scene->getCylinderCount() : 0);
    cylindersOnFrustum = m_stats.visibleCylinderCount;
    drawnCylinders = m_stats.drawnCylinderCount;

    sceneData = {
        {"Screen width", &screenWidth},
        {"Screen height", &screenHeight},
        {"Sphere count", &sphereCount},
        {"Spheres On Frustum", &spheresOnFrustum},
        {"Drawn spheres", &drawnSpheres},
        {"Cylinder count", &cylinderCount},
        {"Cylinders On Frustum", &cylindersOnFrustum},
        {"Drawn cylinders", &drawnCylinders}
    };

    m_window->setupSceneInfoGui("Scene Info", sceneData);
    m_window->setupCameraGui("Camera Info", m_camera.get());
    m_window->setupInputInfoGui("General Input Info");
    
    if (m_benchmark)
    {
        m_window->setupBenchmarkInfoGui("Benchmark", m_benchmark.get());
    }
}

void RenderPipeline::handleInput()
{
    const Input& input = m_window->getInput();

    // Screenshot
    if (input.isKeyDown(Key::F10))
    {
        int checkpoint = m_benchmark ? m_benchmark->getCheckpointID() : 0;
        std::string filename = "media/img/frame_" + std::to_string(checkpoint) + ".bmp";
        takeScreenshot(filename);
    }

    // Start profiling
    if (m_profiler && input.isKeyDown(Key::T))
    {
        int checkpoint = m_benchmark ? m_benchmark->getCheckpointID() : 0;
        m_profiler->startSavingNextFrames(checkpoint);
    }
}

void RenderPipeline::updateCameraAndBenchmark()
{
    m_cameraController->cameraUpdate();
    
    if (m_benchmark)
    {
        m_benchmark->update();
    }
}

void RenderPipeline::updateProfiler()
{
    if (m_profiler)
    {
        int checkpoint = m_benchmark ? m_benchmark->getCheckpointID() : 0;
        m_profiler->updateProfiler(
            checkpoint,
            m_stats.visibleSphereCount,
            m_stats.drawnSphereCount,
            m_stats.visibleCylinderCount,
            m_stats.drawnCylinderCount
        );
    }
}

void RenderPipeline::takeScreenshot(const std::string& filename)
{
    // Implementation would use the canvas's takeScreenshot method
    // This would need access to the renderer's canvas
}
