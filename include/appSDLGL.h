#ifndef _APP_SDL_GL_H_
#define _APP_SDL_GL_H_

#include <iostream>
#include <glad/glad.h>
#include <SDL3/SDL.h>
#include <backends/imgui_impl_opengl3.h>
#include "vis/window.h"

class AppOpenGL : public Window {
public:
    AppOpenGL(std::string& windowTitle, int width, int height, bool shown = true)
        : Window(windowTitle, width, height, shown), m_glContext(nullptr) 
    {
        ////
        createGraphicsContext();
        Window::initImGuiContext();
        initImGui();
        ////
        
        m_startTime = SDL_GetPerformanceCounter();
        m_setupTime -= m_startTime;
    }
    
    ~AppOpenGL() override
    {
        ImGui_ImplOpenGL3_Shutdown();
        Window::shutdownImGuiContext();
        destroyGraphicsContext();
        // Window::~Window();
    }
    
    bool update()
    {
        bool running = true;
        Window::updateTimes();
        Window::checkInputs(running);
        
        // Actualizar pantalla
        renderFrame();
        renderImGui();
        presentFrame();
    
        return running;
    }

    inline SDL_GLContext getContext() { return m_glContext; }

protected:
    

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

    void destroyGraphicsContext() override 
    {
        if (m_glContext) 
        {
            SDL_GL_DestroyContext(m_glContext);
            m_glContext = nullptr;
        }
    }

    void renderFrame() override 
    {
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glBlitFramebuffer(0, 0, m_width, m_height, 0, 0, m_width, m_height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    }

    void presentFrame() override 
    {
        SDL_GL_SwapWindow(m_window);
    }

    void initImGui() override 
    {
        ImGui_ImplSDL3_InitForOpenGL(m_window, m_glContext);
        ImGui_ImplOpenGL3_Init("#version 130");
    }

    void renderImGui() const override 
    {
        ImGui_ImplOpenGL3_NewFrame();
        Window::renderImGui();  
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }

private:
    SDL_GLContext m_glContext;
};

#endif // _APP_SDL_GL_H_