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

class Window
{
  public:
    Window( std::string& title, std::size_t width = 1280, std::size_t height = 720, bool shown = true );

    Window( const Window & )             = delete;
    Window & operator=( const Window & ) = delete;

    Window( Window && other ) noexcept;
    Window & operator=( Window && other ) noexcept;

    virtual ~Window();

    // bool update();

    void resize( std::size_t width, std::size_t height );


    inline SDL_Window *  getHandle() { return m_window; }
    inline uint32_t      getWidth() const { return m_width; }
    inline uint32_t      getHeight() const { return m_height; }
    inline uint64_t      getElapsedTime() const { return m_elapsedTime; }
    inline const Input & getInput() const { return m_input; }

  protected:
    void updateTimes();
    void checkInputs(bool& running);
    virtual void initImGuiContext();
    virtual void setupSceneInfoGui(std::string name, std::unordered_map<std::string, int*>& scene_data);
    virtual void setupCameraGui(std::string name, Camera* camera);
    virtual void setupInputInfoGui(std::string name);
    virtual void setupBenchmarkInfoGui(std::string name, Benchmark* benchmark);
    virtual void startRenderImGui() const;
    virtual void presentRenderImGui() const;
    virtual void shutdownImGuiContext() const;
    virtual bool update() = 0;
    virtual void createGraphicsContext() = 0;
    virtual void destroyGraphicsContext() = 0;
    virtual void initImGui() = 0;
    virtual void renderFrame() = 0;
    virtual void presentFrame() = 0;


    std::string m_title;
    uint32_t    m_width;
    uint32_t    m_height;

    SDL_Window *  m_window    = nullptr;

    Input    m_input {};
    uint64_t m_setupTime = 0;
    uint64_t m_lastTimeStep = 0;
    uint64_t m_startTime = 0;
    uint64_t m_elapsedTime = 0;
    uint64_t m_frame_counter = 0;
    bool     m_isVisible    = true;

    ImGuiIO* io;
    std::unique_ptr<AppUI> m_ui;
};

#endif