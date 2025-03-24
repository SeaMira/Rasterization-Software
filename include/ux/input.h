#ifndef _INPUT_H_
#define _INPUT_H_

#include <optional>
#include <set>

#include <glm/glm.hpp>
#include <SDL3/SDL.h>

/**
 * @enum KeyAction
 * 
 * @brief KeyAction enum for the different actions a key can have.
 */
enum class KeyAction
{
    // Can be hold
    Pressed,

    // Only a single time
    Down,
    Up
};


// see: https://github.com/libsdl-org/SDL/blob/main/include/SDL3/SDL_scancode.h
/**
 * @enum Key
 * 
 * @brief Key enum for the different keys that can be pressed.
 * 
 * The Key enum is used to represent the different keys that can be pressed in the application depending
 * on SDL keys scancode, more info on @link https://github.com/libsdl-org/SDL/blob/main/include/SDL3/SDL_scancode.h SDL 
 * Scancode documentation @endlink.
 */
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

/**
 * @struct Input
 * 
 * @brief Input struct for storing the input of the application.
 * 
 * The Input struct is used to store the input of the application, such as the mouse position, the keys pressed,
 * the keys down, the keys up, the mouse wheel, the mouse clicks and the window size.
 */
struct Input
{

    glm::uvec2 windowSize; ///< the size of the window.
    bool       windowResized = false; ///< flag for checking if the window was resized.

    float deltaTime; ///< the delta time of the last frame.

    glm::ivec2 mousePosition; ///< the position of the mouse.
    glm::ivec2 deltaMousePosition; ///< the delta of the mouse position.

    int32_t deltaMouseWheel; ///< the delta of the mouse wheel scrolling.

    bool doubleLeftClick = false; ///< flag for checking if a double left click was made.

    bool mouseLeftClicked = false; ///< flag for checking if the left mouse button was clicked.
    bool mouseLeftPressed = false; ///< flag for checking if the left mouse button is pressed.

    bool mouseRightClicked = false; ///< flag for checking if the right mouse button was clicked.
    bool mouseRightPressed = false; ///< flag for checking if the right mouse button is pressed.

    bool mouseMiddleClicked = false; ///< flag for checking if the middle mouse button was clicked.
    bool mouseMiddlePressed = false; ///< flag for checking if the middle mouse button is pressed.

    std::set<Key> keysPressed; ///< set of keys that are pressed.
    std::set<Key> keysDown; ///< set of keys that are down.
    std::set<Key> keysUp; ///< set of keys that are released.

    /**
     * @brief checks if a key is pressed.
     * 
     * Checks if a key is pressed, in other words, if is on KeysPressed set.
     * 
     * @param key the key to check.
     * 
     * @return true if the key is pressed, false otherwise.
     */
    bool isKeyPressed( const Key key ) const;

    /**
     * @brief checks if a key is down.
     * 
     * Checks if a key is down, in other words, if is on KeysDown set.
     * 
     * @param key the key to check.
     * 
     * @return true if the key is down, false otherwise.
     */
    bool isKeyDown( const Key key ) const;

    /**
     * @brief checks if a key is up.
     * 
     * Checks if a key is up, in other words, if is on KeysUp set.
     * 
     * @param key the key to check.
     * 
     * @return true if the key is up, false otherwise.
     */
    bool isKeyUp( const Key key ) const;

    /**
     * @brief checks if a key is activated.
     * 
     * Checks if a key is activated, in other words, if is on the set of keys depending on the action.
     * 
     * @param key the key to check.
     * @param action the action to check.
     * 
     * @return true if the key is activated, false otherwise.
     */
    bool isKeyActivated( const Key key, const KeyAction action ) const;

    /**
     * @brief resets the input.
     * 
     * Resets the input so the flags are set to false and the sets are cleared.
     */
    void reset();

    /**
     * @brief destructor for the Input struct.
     * 
     * Destructor for the Input struct, clears the sets of keys.
     */
    ~Input()
    {
        keysPressed = {};
        keysDown = {};
        keysUp = {};

    }
};


#endif // _INPUT_H_