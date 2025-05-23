#ifndef _COMPONENT_H_
#define _COMPONENT_H_

#include <imgui.h>
#include <vector>
#include <string>
#include <memory>
#include "ui/element.h"

/**
 * @class Component
 * @brief Represents a UI component that can contain and render multiple UI elements.
 * 
 * This class allows for the creation of a UI component, which can hold various UI elements and render them. 
 * It provides functionality for managing UI elements and rendering them in the ImGui interface.
 */
class Component {
public:
    /**
     * @brief Constructs a Component with the specified name.
     * @param name The name of the component (used for identification in the UI).
     */
    Component(std::string name) : m_name(name) {}

    /// Virtual destructor for the Component class.
    virtual ~Component() = default;

    /**
     * @brief Renders the UI component by rendering all its associated elements.
     * 
     * This method loops through all the elements contained within the component and renders them.
     */
    virtual void render();
    
    /**
     * @brief Adds an element to the component.
     * @param e A unique pointer to an Element that will be added to the component.
     * 
     * This method moves the element into the component's element list, allowing it to be rendered.
     */
    inline void addElement(std::unique_ptr<Element> e) { elements.push_back(std::move(e));}
    
    /**
     * @brief Gets the name of the component.
     * @return A reference to the name of the component.
     */
    inline std::string& getName() { return m_name; }

protected:
    std::string m_name; ///< The name of the component.
    std::vector<std::unique_ptr<Element>> elements; ///< The list of UI elements associated with the component.
};

#endif // _COMPONENT_H_