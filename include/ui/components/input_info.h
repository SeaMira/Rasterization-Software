#ifndef _INPUT_INFO_H_
#define _INPUT_INFO_H_

#include <vector>
#include "ux/input.h"
#include "ui/component.h"

/**
 * @class InputInfoComponent
 * @brief Represents a UI component for displaying input information.
 * 
 * This class provides functionality for rendering a UI component that displays information about the input, 
 * mainly which keyboard actions are allowed and how to use the camera.
 */
class InputInfoComponent : public Component
{
public:
    /**
     * @brief Constructs a InputInfoComponent with the specified name and input object.
     * 
     * This will create a the interface elements necessary to display the input information.
     * 
     * @param name The name of the component.
     * @param input The input object to display information from.
     */
    InputInfoComponent(std::string name, Input* input);

    /**
     * @brief Action to take a screenshot.
     * 
     * This method is called when the user clicks the "Screenshot" button on the UI.
     */
    void screenshotAction();
private:
    Input* m_input; ///< The input object associated with the component.
};


#endif // _INPUT_INFO_H_