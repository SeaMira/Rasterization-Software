#ifndef _APP_SDL_GL_H_
#define _APP_SDL_GL_H_

#include <iostream>
#include <glad/glad.h>
#include <SDL3/SDL.h>
#include <backends/imgui_impl_opengl3.h>
#include "vis/window.h"

/**
 * @class AppOpenGL
 * 
 * @brief AppOpenGL class extends window to an specific graphic context usage.
 * @extends Window
 */
class AppOpenGL : public Window {
public:
    /** 
     * @brief Class constructor.
     * 
     * @copydoc Window::Window
     * 
     * While inheriting parents constructor it also stablished gl as the graphics context 
     * (for the main app and of the UI).
     * 
     * @see Window::Window For the base constructor.
     */
    AppOpenGL(std::string& title, int width, int height, bool shown = true)
        : Window(title, width, height, shown), m_glContext(nullptr) 
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
     * While inheriting parents destructor it also deals with gl context destructor 
     * (for the main app and the UI).
     * 
     * @see Window::~Window For the base destructor.
     */
    ~AppOpenGL() override
    {
        ImGui_ImplOpenGL3_Shutdown();
        Window::shutdownImGuiContext();
        destroyGraphicsContext();
        // Window::~Window();
    }
    
    /** 
     * @brief Frame renderer.
     * 
     * @copydoc Window::update
     * 
     * While inheriting parents update and input checker it also includes
     * frame render for the main app and the UI.
     * 
     * @see Window::update For the base frame updater.
     */
    bool update()
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
     * @brief Get the GL context handle.
     * 
     * @return GL Context handle.
     */
    inline SDL_GLContext getContext() { return m_glContext; }

protected:
    
    /** 
     * @brief function for starting graphic context.
     * 
     * @copydoc Window::createGraphicsContext
     * 
     * Established gl as the graphics context and associates
     * it with SDL window.
     * 
     * @see Window::createGraphicsContext
     */
    void createGraphicsContext() override 
    {
        m_glContext = SDL_GL_CreateContext(m_window);
        if (!m_glContext) 
        {
            std::cerr << "OpenGL context could not be created! SDL_Error: " << SDL_GetError() << std::endl;
            exit(1);
        }

        SDL_GL_MakeCurrent( m_window, m_glContext );

        if (!gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress)) 
        {
            std::cerr << "Failed to initialize OpenGL loader!" << std::endl;
            exit(1);
        }

        if (GLVersion.major < 4 || (GLVersion.major == 4 && GLVersion.minor < 5)) 
            throw std::runtime_error("OpenGL 4.5 is not supported");
        

    }

    /** 
     * @brief function for destroying graphic context.
     * 
     * @copydoc Window::destroyGraphicsContext
     * 
     * Destroys gl context.
     * 
     * @see Window::destroyGraphicsContext
     */
    void destroyGraphicsContext() override 
    {
        if (m_glContext) 
        {
            SDL_GL_DestroyContext(m_glContext);
            m_glContext = nullptr;
        }
    }

    /** 
     * @brief function for rendering new frame.
     * 
     * @copydoc Window::renderFrame
     * 
     * Updates info on the frambuffer of the viewport.
     * 
     * @see Window::renderFrame
     */
    void renderFrame() override 
    {
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glBlitFramebuffer(0, 0, m_width, m_height, 0, 0, m_width, m_height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    }

    /** 
     * @brief function for presenting new frame.
     * 
     * @copydoc Window::presentFrame
     * 
     * Shows new frame by swapping buffers.
     * 
     * @see Window::presentFrame
     */
    void presentFrame() override 
    {
        SDL_GL_SwapWindow(m_window);
    }

    /** 
     * @brief function for starting ImGUI with gl as graphics context.
     * 
     * @copydoc Window::initImGui
     * 
     * Starts gl context for ImGUI.
     * 
     * @see Window::initImGui
     */
    void initImGui() override 
    {
        ImGui_ImplSDL3_InitForOpenGL(m_window, m_glContext);
        ImGui_ImplOpenGL3_Init("#version 130");
    }

    /** 
     * @brief function for starting new ImGUI frame.
     * 
     * @copydoc Window::startRenderImGui
     * 
     * Starts ImGUI frame with gl context.
     * 
     * @see Window::startRenderImGui
     */
    void startRenderImGui() const override 
    {
        ImGui_ImplOpenGL3_NewFrame();
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
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }

private:
    SDL_GLContext m_glContext; ///< GL context handle. 
};

#endif // _APP_SDL_GL_H_