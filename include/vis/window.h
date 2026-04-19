#ifndef _WINDOW_H_
#define _WINDOW_H_

#include <cstddef>
#include <string>
#include <glad/glad.h>
#include <imgui.h>
#include <backends/imgui_impl_sdl3.h>
#include <unordered_map>

#include "ux/input.h"
#include "ux/camera.h"
// #include "ux/cinematic/benchmark.h"
#include "ui/app_ui.h"

class Benchmark;
/**
 * @class Window
 * 
 * @brief Application window class.
 * 
 * Wrapper for applications events handles. Check inputs, UI, SDL viewport and basic measures.
 * 
 * @param title title to display on the window.
 * @param width width of the window.
 * @param height height of the window.
 * @param shown boolean to know if image should start showing.
 */
class Window
{
  public:
    /**
     * @brief Window class constructor.
     * 
     * Initializes SDL flags, display devices and window. Measures initializing time. 
     * Also prepares UI element.
     * 
     * @param title title to display on the window.
     * @param width width of the window.
     * @param height height of the window.
     * @param shown boolean to know if image should start showing.
     */
    Window( const std::string& title, std::size_t width = 1280, std::size_t height = 720, bool shown = true );

    /**
     * @brief Copy Window operator.
     * 
     * Copy Window object operator has been deleted: not allowed.
     */
    Window( const Window & )             = delete;

    /**
     * @brief Copy Window by asignment operator.
     * 
     * Copy Window object by asignment operator has been deleted: not allowed.
     */
    Window & operator=( const Window & ) = delete;

    /**
     * @brief Move Window object operator.
     * 
     * Move Window object operator.
     */
    Window( Window && other ) noexcept;

    /**
     * @brief Move Window object by asignment operator.
     * 
     * Move Window object by asignment operator.
     */
    Window & operator=( Window && other ) noexcept;

    /**
     * @brief Class destructor.
     * 
     * Takes care of shutting down the window and the subsystem.
     */
    virtual ~Window();

    // bool update();

    /**
     * @brief Resize of window function.
     * 
     * Resizes the window to new values of width and height.
     * 
     * @param width width value of the window to resize.
     * @param height heigth value of the window to resize.
     */
    void resize( std::size_t width, std::size_t height );

    /**
     * @brief Get the SDL window handle.
     */
    inline SDL_Window *  getHandle() { return m_window; }

    /**
     * @brief Get the window width.
     */
    inline uint32_t      getWidth() const { return m_width; }

    /**
     * @brief Get the window height.
     */
    inline uint32_t      getHeight() const { return m_height; }

    /**
     * @brief Get the application elapsed time.
     */
    inline uint64_t      getElapsedTime() const { return m_elapsedTime; }

    /**
     * @brief Get the Input handle.
     */
    inline const Input & getInput() const { return m_input; }

  protected:
    /**
     * @brief Updates application execution times.
     * 
     * Updates last frame delta time, elapsed time and frame counter.
     */
    void updateTimes();

    /**
     * @brief Check new frame inputs.
     * 
     * Checks if mouse or keys were pressed. Some keys can affect the executing condition
     * of the application.
     * 
     * @param running executing condition of the application.
     */
    void checkInputs(bool& running);

    /**
     * @brief initializes ImGUI context for the UI during execution.
     * 
     *  Checks ImGUI version, enables flags and sets some basic style settings.
     */
    virtual void initImGuiContext();

    /**
     * @brief Sets UI Scene Info module.
     * 
     * Adds a UI component for rendering scene info. Recieves name of the component 
     * and its submodules. It's shown as a collapsable menu.
     * 
     * @param name name of the component. 
     * @param scene_data map containing data info in a "category: value" manner.
     */
    virtual void setupSceneInfoGui(std::string name, std::unordered_map<std::string, int*>& scene_data);
    
    /**
     * @brief Sets UI Camera Info module.
     * 
     * Adds a UI component for rendering Camera info. Recieves name of the component 
     * and a pointer to the camera used. It's shown as a collapsable menu.
     * 
     * @param name name of the component. 
     * @param camera pointer to a Camera object from which recieve all data.
     */
    virtual void setupCameraGui(std::string name, Camera* camera);

    /**
     * @brief Sets UI Input Info module.
     * 
     * Adds a UI component for rendering Input info. Recieves name of the component. 
     * It's shown as a collapsable menu. The Input handle is part of the class.
     * 
     * @param name name of the component. 
     */
    virtual void setupInputInfoGui(std::string name);

    /**
     * @brief Sets UI Benchmark Info module.
     * 
     * Adds a UI component for rendering Benchmark info. Recieves name of the component 
     * and a pointer to the benchmark element. It's shown as a collapsable menu.
     * 
     * @param name name of the component. 
     * @param benchmark benchmark element with the current state of the benchmark.
     */
    virtual void setupBenchmarkInfoGui(std::string name, Benchmark* benchmark);

    /**
     * @brief Sets UI Pipeline Config module.
     * 
     * Adds a UI component for configuring pipeline parameters like thresholds and tile settings.
     * 
     * @param name name of the component.
     * @param smallThreshold Reference to the small entity threshold value.
     * @param tileSize Reference to the tile size value.
     * @param maxEntitiesPerTile Reference to max entities per tile.
     * @param tilesX Reference to number of tiles in X.
     * @param tilesY Reference to number of tiles in Y.
     */
    virtual void setupPipelineConfigGui(std::string name, 
                                        int& smallThreshold,
                                        int& tileSize,
                                        int& maxEntitiesPerTile,
                                        int& tilesX,
                                        int& tilesY);

    /** Out-of-core: slider bound to `visibility_threshold` uploaded to CUDA each frame. */
    virtual void setupOocTuningGui(std::string name, float& visibilityThreshold);

    /**
     * @brief Start the ImGUI renderer.
     * 
     * Starts to render a new frame for the ImGUI and displays some basic info 
     * of the application.
     */
    virtual void startRenderImGui() const;

    /**
     * @brief Finishes rendering and presents the results
     */
    virtual void presentRenderImGui() const;

    /**
     * @brief Shuts down ImGUI and destroys its context. 
     */
    virtual void shutdownImGuiContext() const;

    /**
     * @brief Renders new frame. Depends on the context chosen and the UI.
     * 
     * This is a pure virtual function and must be implemented by derived classes.
     */
    virtual bool update() = 0;
    
    /**
     * @brief Creates graphics context depending on the wanted app.
     * 
     * This is a pure virtual function and must be implemented by derived classes.
     */
    virtual void createGraphicsContext() = 0;

    /**
     * @brief Destroys graphics context depending on the wanted app.
     * 
     * This is a pure virtual function and must be implemented by derived classes.
     */
    virtual void destroyGraphicsContext() = 0;

    /**
     * @brief Creates imgui context depending on the wanted app's graphics context.
     * 
     * This is a pure virtual function and must be implemented by derived classes.
     */
    virtual void initImGui() = 0;

    /**
     * @brief Renders a new frame, depends on the chosen graphics context.
     * 
     * This is a pure virtual function and must be implemented by derived classes.
     */
    virtual void renderFrame() = 0;

    /**
     * @brief Presents the new rendered frame, depends on the chosen graphics context.
     * 
     * This is a pure virtual function and must be implemented by derived classes.
     */
    virtual void presentFrame() = 0;


    std::string m_title; ///< title to display on the window.
    uint32_t    m_width; ///< width of the window.
    uint32_t    m_height; ///< height of the window.

    SDL_Window *  m_window    = nullptr; ///< SDL window handle.

    Input    m_input {}; ///< application Input handle.
    uint64_t m_setupTime = 0; ///< time to prepare the application.
    uint64_t m_lastTimeStep = 0; ///< last frame's time of appearance.
    uint64_t m_startTime = 0; ///< application's render loop start time.
    uint64_t m_elapsedTime = 0; ///< application total elapsed time since beggining of render loop.
    uint64_t m_frame_counter = 0; ///< total frame counter
    bool     m_isVisible    = true; ///< visibility of window.

    ImGuiIO* io;  ///< ImGUI context handler.
    std::unique_ptr<AppUI> m_ui; ///< UI application handler.
};

#endif