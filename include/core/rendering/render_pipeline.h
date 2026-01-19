#ifndef _RENDER_PIPELINE_H_
#define _RENDER_PIPELINE_H_

#include <memory>
#include <functional>
#include <vector>

#include "core/interfaces/i_renderer.h"
#include "core/interfaces/i_culling_strategy.h"
#include "core/config/render_config.h"
#include "ux/camera_controller.h"
#include "ux/profiler/profiler.h"
#include "ux/cinematic/benchmark.h"
#include "appSDLGL.h"

// Forward declarations
class Scene;

/**
 * @struct RenderStatistics
 * 
 * @brief Statistics collected during a render frame.
 * 
 * Contains all relevant performance and visibility statistics
 * for a single frame.
 */
struct RenderStatistics
{
    int visibleSphereCount = 0;     ///< Spheres passing frustum culling
    int drawnSphereCount = 0;       ///< Spheres actually rendered
    int visibleCylinderCount = 0;   ///< Cylinders passing frustum culling
    int drawnCylinderCount = 0;     ///< Cylinders actually rendered
    int currentCheckpoint = 0;      ///< Current benchmark checkpoint
    float frameTime = 0.0f;         ///< Frame time in milliseconds
};

/**
 * @class RenderPipeline
 * 
 * @brief Orchestrates the complete rendering pipeline.
 * 
 * Manages the integration of rendering, camera control, profiling,
 * and benchmarking. Follows the Facade Pattern by providing a
 * simplified interface to the complex subsystem.
 * 
 * Also follows the Single Responsibility Principle (SRP) by
 * handling only the orchestration of rendering components.
 */
class RenderPipeline
{
public:
    /**
     * @brief Callback type for frame updates.
     * 
     * @param stats Current frame statistics.
     * @return True to continue running, false to exit.
     */
    using FrameCallback = std::function<bool(const RenderStatistics&)>;

    /**
     * @brief Constructs the render pipeline.
     * 
     * @param config Rendering configuration.
     */
    explicit RenderPipeline(const RenderConfig& config);

    /**
     * @brief Destructor.
     */
    ~RenderPipeline();

    // Prevent copying
    RenderPipeline(const RenderPipeline&) = delete;
    RenderPipeline& operator=(const RenderPipeline&) = delete;

    // Allow moving
    RenderPipeline(RenderPipeline&&) noexcept;
    RenderPipeline& operator=(RenderPipeline&&) noexcept;

    /**
     * @brief Initializes the pipeline with a scene.
     * 
     * @param scene Scene to render.
     */
    void initialize(std::unique_ptr<Scene> scene);

    /**
     * @brief Sets the renderer implementation.
     * 
     * @param renderer Renderer to use.
     */
    void setRenderer(std::unique_ptr<IRenderer> renderer);

    /**
     * @brief Enables profiling with specified output files.
     * 
     * @param frameTimesFile Path for frame times CSV.
     * @param processTimesFile Path for process times CSV.
     * @param timerDuration Duration to record in seconds.
     */
    void enableProfiling(const std::string& frameTimesFile,
                         const std::string& processTimesFile,
                         double timerDuration = 48.0);

    /**
     * @brief Enables benchmarking with checkpoints.
     * 
     * @param checkpoints Vector of position/target pairs.
     */
    void enableBenchmark(const std::vector<std::pair<glm::vec3, glm::vec3>>& checkpoints);

    /**
     * @brief Sets a callback to be called each frame.
     * 
     * @param callback Function to call with frame statistics.
     */
    void setFrameCallback(FrameCallback callback);

    /**
     * @brief Runs the main render loop.
     * 
     * Continues until the window is closed or callback returns false.
     */
    void run();

    /**
     * @brief Executes a single frame.
     * 
     * @return True if should continue, false to exit.
     */
    bool executeFrame();

    /**
     * @brief Gets the current frame statistics.
     * 
     * @return Current render statistics.
     */
    const RenderStatistics& getStatistics() const { return m_stats; }

    /**
     * @brief Gets the window handle.
     * 
     * @return Reference to the application window.
     */
    AppOpenGL& getWindow() { return *m_window; }

    /**
     * @brief Gets the camera.
     * 
     * @return Reference to the camera.
     */
    Camera& getCamera() { return *m_camera; }

    /**
     * @brief Gets the camera controller.
     * 
     * @return Reference to the camera controller.
     */
    CameraController& getCameraController() { return *m_cameraController; }

    /**
     * @brief Gets the benchmark (if enabled).
     * 
     * @return Pointer to benchmark, or nullptr if not enabled.
     */
    Benchmark* getBenchmark() { return m_benchmark.get(); }

    /**
     * @brief Gets the profiler (if enabled).
     * 
     * @return Pointer to profiler, or nullptr if not enabled.
     */
    Profiler* getProfiler() { return m_profiler.get(); }

    /**
     * @brief Gets the scene.
     * 
     * @return Reference to the scene.
     */
    Scene& getScene() { return *m_scene; }

    /**
     * @brief Takes a screenshot.
     * 
     * @param filename Output filename.
     */
    void takeScreenshot(const std::string& filename);

private:
    /**
     * @brief Sets up the UI components.
     */
    void setupUI();

    /**
     * @brief Handles input for the current frame.
     */
    void handleInput();

    /**
     * @brief Updates the camera and benchmark.
     */
    void updateCameraAndBenchmark();

    /**
     * @brief Updates the profiler with current stats.
     */
    void updateProfiler();

private:
    RenderConfig m_config;                          ///< Render configuration
    RenderStatistics m_stats;                       ///< Current frame statistics
    
    std::unique_ptr<AppOpenGL> m_window;            ///< Application window
    std::unique_ptr<Camera> m_camera;               ///< Scene camera
    std::unique_ptr<CameraController> m_cameraController; ///< Camera controller
    std::unique_ptr<Scene> m_scene;                 ///< Scene to render
    std::unique_ptr<IRenderer> m_renderer;          ///< Active renderer
    
    std::unique_ptr<Profiler> m_profiler;           ///< Optional profiler
    std::unique_ptr<Benchmark> m_benchmark;         ///< Optional benchmark
    
    FrameCallback m_frameCallback;                  ///< Optional frame callback
    bool m_initialized = false;                     ///< Initialization flag
};

#endif // _RENDER_PIPELINE_H_
