#include "ux/input.h"

bool Input::isKeyPressed( const Key key ) const { return keysPressed.find( key ) != keysPressed.end(); }
bool Input::isKeyDown( const Key key ) const { return keysDown.find( key ) != keysDown.end(); }
bool Input::isKeyUp( const Key key ) const { return keysUp.find( key ) != keysUp.end(); }
bool Input::isKeyActivated( const Key key, const KeyAction action ) const
{
    switch ( action )
    {
    case KeyAction::Down: return isKeyDown( key );
    case KeyAction::Up: return isKeyUp( key );
    case KeyAction::Pressed: return isKeyPressed( key );
    }
    return false;
}

void Input::reset()
{
    doubleLeftClick = false;

    mouseLeftClicked   = false;
    mouseRightClicked  = false;
    mouseMiddleClicked = false;

    windowResized      = false;
    deltaMouseWheel    = 0;
    deltaMousePosition = {};
    keysDown           = {};
    keysUp             = {};
}