#ifndef _APP_SDL_RENDERER_H_
#define _APP_SDL_RENDERER_H_

#include <iostream>
#include <glad/glad.h>
#include <SDL3/SDL.h>
#include <backends/imgui_impl_sdlrenderer3.h>
#include "vis/window.h"


/**
 * @class AppRenderer
 * 
 * @brief AppRenderer class extends window to an specific graphic context usage.
 * @extends Window
 */
class AppRenderer : public Window {
public:
    /** 
     * @brief Class constructor.
     * 
     * @copydoc Window::Window
     * 
     * While inheriting parents constructor it also stablished SDL and its renderer (on a texture) as its
     * graphic context (for the main app and the UI).
     * 
     * @see Window::Window For the base constructor.
     */
    AppRenderer(std::string& windowTitle, int width, int height, bool shown = true)
        : Window(windowTitle, width, height, shown), m_renderer(nullptr), m_texture(nullptr) 
    {
        ////
        createGraphicsContext();
        Window::initImGuiContext();
        initImGui();
        ////

        m_startTime = SDL_GetPerformanceCounter();
        m_setupTime -= m_startTime;
    }

    /** 
     * @brief Class destructor.
     * 
     * @copydoc Window::~Window
     * 
     * While inheriting parents destructor it also deals with SDL renderer destructor 
     * (for the main app and the UI).
     * 
     * @see Window::~Window For the base destructor.
     */
    ~AppRenderer() override
    {
        ImGui_ImplSDLRenderer3_Shutdown();
        Window::shutdownImGuiContext();
        destroyGraphicsContext();
        // Window::~Window();
    }

    /** 
     * @brief Texture updater.
     * 
     * Loads framebuffer data on the texture to be displayed.
     */
    void updateTexture(std::vector<uint32_t>& framebuffer)
    {
        SDL_UpdateTexture(m_texture, nullptr, framebuffer.data(), m_width * sizeof(uint32_t));
    }
    
    /** 
     * @brief Frame renderer.
     * 
     * @copydoc Window::update
     * 
     * While inheriting parents update and input checker it also includes
     * frame render for the main app and the UI with SDL Renderer.
     * 
     * @see Window::update For the base frame updater.
     */
    bool update() override
    {
        bool running = true;
        Window::updateTimes();
        Window::checkInputs(running);
    
        // Actualizar pantalla
        renderFrame();
        startRenderImGui();
        m_ui->render();
        presentRenderImGui();
        presentFrame();
    
        return running;
    }
    
    /** 
     * @brief UI Scene Info module setup.
     * 
     * @copydoc Window::setupSceneInfoGui
     */
    void setupSceneInfoGui(std::string name, std::unordered_map<std::string, int*>& scene_data) override
    {
        Window::setupSceneInfoGui(name, scene_data);
    }

    /** 
     * @brief UI Camera Info module setup.
     * 
     * @copydoc Window::setupCameraGui
     */
    void setupCameraGui(std::string name, Camera* camera) override
    {
        Window::setupCameraGui(name, camera);
    }
    
    /** 
     * @brief UI Input Info module setup.
     * 
     * @copydoc Window::setupInputInfoGui
     */
    void setupInputInfoGui(std::string name) override
    {
        Window::setupInputInfoGui(name);
    }
    
    /** 
     * @brief UI Benchmark Info module setup.
     * 
     * @copydoc Window::setupBenchmarkInfoGui
     */
    void setupBenchmarkInfoGui(std::string name, Benchmark* benchmark) override
    {
        Window::setupBenchmarkInfoGui(name, benchmark);
    }

    /**
     * @brief Get the SDL Renderer handle.
     * 
     * @return SDL Renderer handle.
     */
    inline SDL_Renderer* getRenderer() const { return m_renderer; }
protected:
    

    /** 
    * @brief function for starting graphic context.
    * 
    * @copydoc Window::createGraphicsContext
    * 
    * Established SDL Renderer showing a texture as the graphics context and associates
    * it with SDL window.
    * 
    * @see Window::createGraphicsContext
    */
    void createGraphicsContext() override 
    {
        m_renderer = SDL_CreateRenderer(m_window, nullptr);
        if (!m_renderer) {
            std::cerr << "SDL_Renderer could not be created! SDL_Error: " << SDL_GetError() << std::endl;
            exit(1);
        }
        m_texture = SDL_CreateTexture(m_renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, m_width, m_height);
        if (!m_texture) {
            std::cerr << "SDL_Texture could not be created! SDL_Error: " << SDL_GetError() << std::endl;
            exit(1);
        }
    }

    /** 
     * @brief function for destroying graphic context.
     * 
     * @copydoc Window::destroyGraphicsContext
     * 
     * Destroys renderer and texture context.
     * 
     * @see Window::destroyGraphicsContext
     */
    void destroyGraphicsContext() override 
    {
        if (m_texture) 
        {
            SDL_DestroyTexture(m_texture);
            m_texture = nullptr;
        }
        if (m_renderer) 
        {
            SDL_DestroyRenderer(m_renderer);
            m_renderer = nullptr;
        }
    }


    /** 
     * @brief function for rendering new frame.
     * 
     * @copydoc Window::renderFrame
     * 
     * Updates info on the renderer with the texture's info.
     * 
     * @see Window::renderFrame
     */
    void renderFrame() override 
    {
        SDL_RenderClear(m_renderer);
        SDL_RenderTexture(m_renderer, m_texture, nullptr, nullptr);
    }

    /** 
     * @brief function for presenting new frame.
     * 
     * @copydoc Window::presentFrame
     * 
     * Shows new frame by presenting renderer info.
     * 
     * @see Window::presentFrame
     */
    void presentFrame() override 
    {
        SDL_RenderPresent(m_renderer);
    }

    /** 
     * @brief function for starting ImGUI with just SDL Renderer as graphics context.
     * 
     * @copydoc Window::initImGui
     * 
     * Starts SDL Renderer context for ImGUI.
     * 
     * @see Window::initImGui
     */
    void initImGui() override 
    {
        ImGui_ImplSDL3_InitForSDLRenderer(m_window, m_renderer);
        ImGui_ImplSDLRenderer3_Init(m_renderer);
    }

    /** 
     * @brief function for starting new ImGUI frame.
     * 
     * @copydoc Window::startRenderImGui
     * 
     * Starts ImGUI frame with SDL Renderer context.
     * 
     * @see Window::startRenderImGui
     */
    void startRenderImGui() const override 
    {
        ImGui_ImplSDLRenderer3_NewFrame();
        Window::startRenderImGui();  
    }

    /** 
     * @brief function for presenting new ImGUI frame.
     * 
     * @copydoc Window::presentRenderImGui
     * 
     * Presents new frame on UI.
     * 
     * @see Window::presentRenderImGui
     */
    virtual void presentRenderImGui() const override
    {
        Window::presentRenderImGui();
        ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), m_renderer);
    }

private:
    SDL_Renderer* m_renderer; ///< SDL Renderer context handle. 
    SDL_Texture* m_texture; ///< SDL Texture handle. 
};

#endif // _APP_SDL_RENDERER_H_