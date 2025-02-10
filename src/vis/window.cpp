#include <iostream>
#include <SDL3/SDL.h>

#include "vis/window.h"

    

Window::Window( std::string& title, std::size_t width, std::size_t height, bool shown ) :
    m_title( std::move( title ) ), m_width( width ), m_height( height )
{
    if ( SDL_Init( SDL_INIT_VIDEO ) < 0 )
    {
        std::cout << "alo" << std::endl;
        throw std::runtime_error( SDL_GetError() );
    }
    SDL_GL_SetAttribute( SDL_GL_CONTEXT_FLAGS, 0 );
    SDL_GL_SetAttribute( SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE );
    SDL_GL_SetAttribute( SDL_GL_CONTEXT_MAJOR_VERSION, 4 );
    SDL_GL_SetAttribute( SDL_GL_CONTEXT_MINOR_VERSION, 5 );
    SDL_GL_SetAttribute( SDL_GL_DOUBLEBUFFER, 1 );
    SDL_GL_SetAttribute( SDL_GL_DEPTH_SIZE, 24 );
    SDL_GL_SetAttribute( SDL_GL_STENCIL_SIZE, 8 );

    int num_displays = 0;
    SDL_DisplayID* displays = SDL_GetDisplays(&num_displays);
    if (displays == nullptr || num_displays == 0) {
        std::cerr << "No screens found: " << SDL_GetError() << std::endl;
        SDL_Quit();
    }

    for (int i = 0; i < num_displays; ++i) {
        SDL_DisplayID display_id = displays[i];
        const SDL_DisplayMode* mode = SDL_GetCurrentDisplayMode(display_id);
        if (mode != nullptr) {
            std::cout << "Screen " << i << " - Resolution: " << mode->w << "x" << mode->h
                      << ", Refresh rate: " << mode->refresh_rate << " Hz" << std::endl;
        } else {
            std::cerr << "No screen mode found" << i << ": " << SDL_GetError() << std::endl;
        }
    }

    const uint32_t visibilityFlags = shown ? 
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE : 
        SDL_WINDOW_HIDDEN | SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE;

    m_window
        = SDL_CreateWindow( m_title.c_str(),
                            static_cast<int>( width ),
                            static_cast<int>( height ),
                            visibilityFlags);

    if ( !m_window )
        throw std::runtime_error( SDL_GetError() );

    m_glContext = SDL_GL_CreateContext( m_window );
    if ( !m_glContext )
        throw std::runtime_error( SDL_GetError() );

    SDL_GL_MakeCurrent( m_window, m_glContext );

    if (!gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress)) {
        throw std::runtime_error("Failed to initialize GLAD");
    }

    if (GLVersion.major < 4 || (GLVersion.major == 4 && GLVersion.minor < 5)) {
        throw std::runtime_error("OpenGL 4.5 is not supported");
    }
    m_startTime = SDL_GetPerformanceCounter();
}

Window::Window( Window && other ) noexcept
{
    std::swap( m_title, other.m_title );
    std::swap( m_width, other.m_width );
    std::swap( m_height, other.m_height );
    std::swap( m_window, other.m_window );
    std::swap( m_glContext, other.m_glContext );
    std::swap( m_input, other.m_input );
    std::swap( m_lastTimeStep, other.m_lastTimeStep );
    std::swap( m_isVisible, other.m_isVisible );
}

Window & Window::operator=( Window && other ) noexcept
{
    std::swap( m_title, other.m_title );
    std::swap( m_width, other.m_width );
    std::swap( m_height, other.m_height );
    std::swap( m_window, other.m_window );
    std::swap( m_glContext, other.m_glContext );
    std::swap( m_input, other.m_input );
    std::swap( m_lastTimeStep, other.m_lastTimeStep );
    std::swap( m_isVisible, other.m_isVisible );

    return *this;
}

Window::~Window()
{
    if ( m_glContext )
        SDL_GL_DestroyContext( m_glContext );

    if ( m_window )
        SDL_DestroyWindow( m_window );

    SDL_Quit();
}

static Key toKey( SDL_Scancode scanCode );

bool Window::update()
{
    m_input.reset();

    // Based on ImGui
    // https://github.com/ocornut/imgui/blob/master/backends/imgui_impl_sdl.cpp#L557
    static const uint64_t sdlFrequency = SDL_GetPerformanceFrequency();
    const uint64_t        now          = SDL_GetPerformanceCounter();
    m_input.deltaTime = static_cast<float>( static_cast<double>( now - m_lastTimeStep ) / sdlFrequency );
    m_lastTimeStep    = now;
    m_elapsedTime    = (now - m_startTime)/ 100000.0f;

    bool      running = true;
    SDL_Event windowEvent;

    while ( SDL_PollEvent( &windowEvent ) )
    {
        switch ( windowEvent.type )
        {
        case SDL_EVENT_QUIT: running = false; break;
        case SDL_EVENT_MOUSE_WHEEL:
        {
            m_input.deltaMouseWheel = windowEvent.wheel.y;
            break;
        }
        case SDL_EVENT_MOUSE_MOTION:
        {
            m_input.deltaMousePosition.x = windowEvent.motion.xrel;
            m_input.deltaMousePosition.y = windowEvent.motion.yrel;
            m_input.mousePosition.x      = windowEvent.motion.x;
            m_input.mousePosition.y      = windowEvent.motion.y;
            break;
        }
        case SDL_EVENT_KEY_DOWN:
        {
            Key key = toKey( windowEvent.key.scancode );
            if (key == Key::Escape) return false;
            m_input.keysPressed.emplace( key );
            m_input.keysDown.emplace( key );
            break;
        }
        case SDL_EVENT_KEY_UP:
        {
            Key key = toKey( windowEvent.key.scancode );
            m_input.keysPressed.erase( key );
            m_input.keysUp.emplace( key );
            break;
        }
        case SDL_EVENT_MOUSE_BUTTON_DOWN:
        {
            switch ( windowEvent.button.button )
            {
            case SDL_BUTTON_LEFT:
            {
                m_input.mouseLeftPressed = true;
                m_input.mouseLeftClicked = true;
                m_input.doubleLeftClick  = windowEvent.button.clicks == 2;
                break;
            }
            case SDL_BUTTON_RIGHT:
                m_input.mouseRightPressed = true;
                m_input.mouseRightClicked = true;
                break;
            case SDL_BUTTON_MIDDLE:
                m_input.mouseMiddlePressed = true;
                m_input.mouseMiddleClicked = true;
                break;
            }
            break;
        }
        case SDL_EVENT_MOUSE_BUTTON_UP:
        {
            switch ( windowEvent.button.button )
            {
            case SDL_BUTTON_LEFT: m_input.mouseLeftPressed = false; break;
            case SDL_BUTTON_RIGHT: m_input.mouseRightPressed = false; break;
            case SDL_BUTTON_MIDDLE: m_input.mouseMiddlePressed = false; break;
            }
            break;
        }
        case SDL_EVENT_WINDOW_RESIZED:
        {
            m_width  = windowEvent.window.data1;
            m_height = windowEvent.window.data2;

            m_input.windowSize    = { m_width, m_height };
            m_input.windowResized = true;
            break;
        }
        }
    }

    return running;
}

void Window::resize( std::size_t width, std::size_t height ) { SDL_SetWindowSize( m_window, width, height ); }

static Key toKey( SDL_Scancode scanCode )
{
    switch ( scanCode )
    {
    case SDL_SCANCODE_A: return Key::A;
    case SDL_SCANCODE_B: return Key::B;
    case SDL_SCANCODE_C: return Key::C;
    case SDL_SCANCODE_D: return Key::D;
    case SDL_SCANCODE_E: return Key::E;
    case SDL_SCANCODE_F: return Key::F;
    case SDL_SCANCODE_G: return Key::G;
    case SDL_SCANCODE_H: return Key::H;
    case SDL_SCANCODE_I: return Key::I;
    case SDL_SCANCODE_J: return Key::J;
    case SDL_SCANCODE_K: return Key::K;
    case SDL_SCANCODE_L: return Key::L;
    case SDL_SCANCODE_M: return Key::M;
    case SDL_SCANCODE_N: return Key::N;
    case SDL_SCANCODE_O: return Key::O;
    case SDL_SCANCODE_P: return Key::P;
    case SDL_SCANCODE_Q: return Key::Q;
    case SDL_SCANCODE_R: return Key::R;
    case SDL_SCANCODE_S: return Key::S;
    case SDL_SCANCODE_T: return Key::T;
    case SDL_SCANCODE_U: return Key::U;
    case SDL_SCANCODE_V: return Key::V;
    case SDL_SCANCODE_W: return Key::W;
    case SDL_SCANCODE_X: return Key::X;
    case SDL_SCANCODE_Y: return Key::Y;
    case SDL_SCANCODE_Z: return Key::Z;

    case SDL_SCANCODE_RETURN: return Key::Return;
    case SDL_SCANCODE_ESCAPE: return Key::Escape;
    case SDL_SCANCODE_BACKSPACE: return Key::BackSpace;
    case SDL_SCANCODE_TAB: return Key::Tab;
    case SDL_SCANCODE_SPACE: return Key::Space;

    case SDL_SCANCODE_F1: return Key::F1;
    case SDL_SCANCODE_F2: return Key::F2;
    case SDL_SCANCODE_F3: return Key::F3;
    case SDL_SCANCODE_F4: return Key::F4;
    case SDL_SCANCODE_F5: return Key::F5;
    case SDL_SCANCODE_F6: return Key::F6;
    case SDL_SCANCODE_F7: return Key::F7;
    case SDL_SCANCODE_F8: return Key::F8;
    case SDL_SCANCODE_F9: return Key::F9;
    case SDL_SCANCODE_F10: return Key::F10;
    case SDL_SCANCODE_F11: return Key::F11;
    case SDL_SCANCODE_F12: return Key::F12;

    case SDL_SCANCODE_RIGHT: return Key::Right;
    case SDL_SCANCODE_LEFT: return Key::Left;
    case SDL_SCANCODE_DOWN: return Key::Down;
    case SDL_SCANCODE_UP: return Key::Up;

    case SDL_SCANCODE_LCTRL: return Key::LCtrl;
    case SDL_SCANCODE_LSHIFT: return Key::LShift;
    case SDL_SCANCODE_LALT: return Key::LAlt; /**< alt, option */
    case SDL_SCANCODE_LGUI: return Key::LGui; /**< windows, command (apple), meta */
    case SDL_SCANCODE_RCTRL: return Key::RCtrl;
    case SDL_SCANCODE_RSHIFT: return Key::RShift;
    case SDL_SCANCODE_RALT: return Key::RAlt; /**< alt gr, option */
    case SDL_SCANCODE_RGUI: return Key::RGui; /**< windows, command (apple), meta */
    }

    return Key::Unknown;
}



