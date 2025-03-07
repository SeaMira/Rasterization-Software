#ifndef _APP_SDL_RENDERER_H_
#define _APP_SDL_RENDERER_H_

#include <iostream>
#include <glad/glad.h>
#include <SDL3/SDL.h>
#include <backends/imgui_impl_sdlrenderer3.h>
#include "vis/window.h"

class AppRenderer : public Window {
public:
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

    ~AppRenderer() override
    {
        ImGui_ImplSDLRenderer3_Shutdown();
        Window::shutdownImGuiContext();
        destroyGraphicsContext();
        // Window::~Window();
    }

    void updateTexture(std::vector<uint32_t>& framebuffer)
    {
        SDL_UpdateTexture(m_texture, nullptr, framebuffer.data(), m_width * sizeof(uint32_t));
    }
    
    bool update() override
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

    inline SDL_Renderer* getRenderer() const { return m_renderer; }
protected:
    

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



    void renderFrame() override 
    {
        SDL_RenderClear(m_renderer);
        SDL_RenderTexture(m_renderer, m_texture, nullptr, nullptr);
    }

    void presentFrame() override 
    {
        SDL_RenderPresent(m_renderer);
    }

    void initImGui() override 
    {
        ImGui_ImplSDL3_InitForSDLRenderer(m_window, m_renderer);
        ImGui_ImplSDLRenderer3_Init(m_renderer);
    }

    void renderImGui() const override 
    {
        ImGui_ImplSDLRenderer3_NewFrame();
        Window::renderImGui();  
        ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), m_renderer);
    }

private:
    SDL_Renderer* m_renderer;
    SDL_Texture* m_texture;
};

#endif // _APP_SDL_RENDERER_H_