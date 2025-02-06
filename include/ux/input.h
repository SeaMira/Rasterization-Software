#ifndef _INPUT_H_
#define _INPUT_H_

#include <optional>
#include <set>

#include <glm/vec2.hpp>
#include <SDL3/SDL.h>

enum class KeyAction
{
    // Can be hold
    Pressed,

    // Only a single time
    Down,
    Up
};

// see: https://github.com/libsdl-org/SDL/blob/main/include/SDL3/SDL_scancode.h
enum class Key
{
    Unknown = SDL_SCANCODE_UNKNOWN,

    A = SDL_SCANCODE_A,
    B = SDL_SCANCODE_B,
    C = SDL_SCANCODE_C,
    D = SDL_SCANCODE_D,
    E = SDL_SCANCODE_E,
    F = SDL_SCANCODE_F,
    G = SDL_SCANCODE_G,
    H = SDL_SCANCODE_H,
    I = SDL_SCANCODE_I,
    J = SDL_SCANCODE_J,
    K = SDL_SCANCODE_K,
    L = SDL_SCANCODE_L,
    M = SDL_SCANCODE_M,
    N = SDL_SCANCODE_N,
    O = SDL_SCANCODE_O,
    P = SDL_SCANCODE_P,
    Q = SDL_SCANCODE_Q,
    R = SDL_SCANCODE_R,
    S = SDL_SCANCODE_S,
    T = SDL_SCANCODE_T,
    U = SDL_SCANCODE_U,
    V = SDL_SCANCODE_V,
    W = SDL_SCANCODE_W,
    X = SDL_SCANCODE_X,
    Y = SDL_SCANCODE_Y,
    Z = SDL_SCANCODE_Z,

    Return = SDL_SCANCODE_RETURN,
    Escape = SDL_SCANCODE_ESCAPE,
    BackSpace = SDL_SCANCODE_BACKSPACE,
    Tab = SDL_SCANCODE_TAB,
    Space = SDL_SCANCODE_SPACE,

    F1 = SDL_SCANCODE_F1,
    F2 = SDL_SCANCODE_F2,
    F3 = SDL_SCANCODE_F3,
    F4 = SDL_SCANCODE_F4,
    F5 = SDL_SCANCODE_F5,
    F6 = SDL_SCANCODE_F6,
    F7 = SDL_SCANCODE_F7,
    F8 = SDL_SCANCODE_F8,
    F9 = SDL_SCANCODE_F9,
    F10 = SDL_SCANCODE_F10,
    F11 = SDL_SCANCODE_F11,
    F12 = SDL_SCANCODE_F12,

    Right = SDL_SCANCODE_RIGHT,
    Left = SDL_SCANCODE_LEFT,
    Down = SDL_SCANCODE_DOWN,
    Up = SDL_SCANCODE_UP,

    LCtrl = SDL_SCANCODE_LCTRL,
    LShift = SDL_SCANCODE_LSHIFT,
    LAlt = SDL_SCANCODE_LALT, /**< alt, option */
    LGui = SDL_SCANCODE_LGUI, /**< windows, command (apple), meta */
    RCtrl = SDL_SCANCODE_RCTRL,
    RShift = SDL_SCANCODE_RSHIFT,
    RAlt = SDL_SCANCODE_RALT, /**< alt gr, option */
    RGui = SDL_SCANCODE_RGUI, /**< windows, command (apple), meta */
};

struct Input
{
    glm::uvec2 windowSize;
    bool       windowResized = false;

    float deltaTime;

    glm::ivec2 mousePosition;
    glm::ivec2 deltaMousePosition;

    int32_t deltaMouseWheel;

    bool doubleLeftClick = false;

    bool mouseLeftClicked = false;
    bool mouseLeftPressed = false;

    bool mouseRightClicked = false;
    bool mouseRightPressed = false;

    bool mouseMiddleClicked = false;
    bool mouseMiddlePressed = false;

    std::set<Key> keysPressed;
    std::set<Key> keysDown;
    std::set<Key> keysUp;

    bool isKeyPressed( const Key key ) const;
    bool isKeyDown( const Key key ) const;
    bool isKeyUp( const Key key ) const;

    bool isKeyActivated( const Key key, const KeyAction action ) const;

    void reset();
};


#endif // _INPUT_H_