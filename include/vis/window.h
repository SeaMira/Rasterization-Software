#ifndef _WINDOW_H_
#define _WINDOW_H_

#include <cstddef>
#include <string>
#include <glad/glad.h>

#include "ux/input.h"

class Window
{
  public:
    Window( std::string& title, std::size_t width = 1280, std::size_t height = 720, bool shown = true );

    Window( const Window & )             = delete;
    Window & operator=( const Window & ) = delete;

    Window( Window && other ) noexcept;
    Window & operator=( Window && other ) noexcept;

    ~Window();

    bool update();

    void resize( std::size_t width, std::size_t height );

    inline SDL_Window *  getHandle() { return m_window; }
    inline SDL_GLContext getContext() { return m_glContext; }
    inline uint32_t      getWidth() const { return m_width; }
    inline uint32_t      getHeight() const { return m_height; }
    inline uint64_t      getElapsedTime() const { return m_elapsedTime; }
    inline const Input & getInput() const { return m_input; }

  private:
    std::string m_title;
    uint32_t    m_width;
    uint32_t    m_height;

    SDL_Window *  m_window    = nullptr;
    SDL_GLContext m_glContext = nullptr;

    Input    m_input {};
    uint64_t m_lastTimeStep = 0;
    uint64_t m_startTime = 0;
    uint64_t m_elapsedTime = 0;
    bool     m_isVisible    = true;
};

#endif