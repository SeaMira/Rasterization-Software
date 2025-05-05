#ifndef _ELEMENT_H_
#define _ELEMENT_H_
#include <imgui.h>
#include <glm/glm.hpp>
#include <string>
#include <functional> 


/**
 * @class Element
 * @brief Abstract base class for all UI elements in the application.
 * 
 * This class serves as a base for all UI elements that can be rendered in the ImGui interface.
 * It defines a pure virtual function `render()` that must be implemented by derived classes.
 */
class Element {
public:
    /// Virtual destructor for the Element class.
    virtual ~Element() = default;

    /**
     * @brief Pure virtual method to render the element.
     * 
     * Derived classes must implement this method to render their specific UI component.
     */
    virtual void render() = 0;
};


/**
 * @class SliderElement
 * @brief A UI element representing a slider with float values.
 * 
 * This element allows the user to adjust a float value between a minimum and maximum range.
 */
class SliderElement: public Element
{   
public:
    /**
     * @brief Constructs a SliderElement with the specified parameters.
     * @param label The label to display next to the slider.
     * @param value A reference to the float value that the slider will modify.
     * @param minValue The minimum value of the slider.
     * @param maxValue The maximum value of the slider.
     */
    SliderElement(std::string label, float& value, float minValue, float maxValue)
        : label(label), value(&value), minValue(minValue), maxValue(maxValue) {}

    /// Renders the slider using ImGui.
    void render() override 
    {
    ImGui::SliderFloat(label.c_str(), value, minValue, maxValue);
    }
private:
    std::string label; ///< string: The label of the slider.
    float* value;      ///< float*: The float value that the slider modifies.
    float minValue;    ///< float: The minimum value of the slider.
    float maxValue;    ///< float: The maximum value of the slider.
};

/**
 * @class SliderElementTopBounded
 * @brief A slider element where the maximum value is dynamically set by a reference.
 * 
 * Similar to `SliderElement`, but the maximum value can be modified externally.
 */
class SliderElementTopBounded: public Element
{
public:
    /**
     * @brief Constructs a SliderElementTopBounded with the specified parameters.
     * @param label The label to display next to the slider.
     * @param value A reference to the float value that the slider will modify.
     * @param minValue The minimum value of the slider.
     * @param maxValue A reference to the maximum value of the slider.
     */
    SliderElementTopBounded(std::string label, float& value, float minValue, float& maxValue)
        : label(label), value(&value), minValue(minValue), maxValue(&maxValue) {}

    /// Renders the slider using ImGui with a dynamic maximum value.
    void render() override 
    {
    ImGui::SliderFloat(label.c_str(), value, minValue, *maxValue);
    }
private:
    std::string label; ///< string: The label of the slider.
    float* value;      ///< float*: The float value that the slider modifies.
    float minValue;    ///< float: The minimum value of the slider.
    float* maxValue;   ///< float*: A reference to the maximum value of the slider.
};

/**
 * @class SliderElementTopBoundedI
 * @brief A slider element for ints where the maximum value is dynamically set by a reference.
 * 
 * Similar to `SliderElementTopBounded`, but for integers.
 */
class SliderElementTopBoundedI: public Element
{
public:
    /**
     * @brief Constructs a SliderElementTopBounded with the specified parameters.
     * @param label The label to display next to the slider.
     * @param value A reference to the int value that the slider will modify.
     * @param minValue The minimum value of the slider.
     * @param maxValue A reference to the maximum value of the slider.
     */
    SliderElementTopBoundedI(std::string label, int& value, int minValue, int& maxValue)
        : label(label), value(&value), minValue(minValue), maxValue(&maxValue) {}

    /// Renders the slider using ImGui with a dynamic maximum value.
    void render() override 
    {
    ImGui::SliderInt(label.c_str(), value, minValue, *maxValue);
    }
private:
    std::string label; ///< string: The label of the slider.
    int* value;      ///< int*: The int value that the slider modifies.
    int minValue;    ///< int: The minimum value of the slider.
    int* maxValue;   ///< int*: A reference to the maximum value of the slider.
};

/**
 * @class SliderElementLowBounded
 * @brief A slider element where the minimum value is dynamically set by a reference.
 * 
 * Similar to `SliderElement`, but the minimum value can be modified externally.
 */
class SliderElementLowBounded: public Element
{
public:
    /**
     * @brief Constructs a SliderElementLowBounded with the specified parameters.
     * @param label The label to display next to the slider.
     * @param value A reference to the float value that the slider will modify.
     * @param minValue A reference to the minimum value of the slider.
     * @param maxValue The maximum value of the slider.
     */
    SliderElementLowBounded(std::string label, float& value, float& minValue, float maxValue)
        : label(label), value(&value), minValue(&minValue), maxValue(maxValue) {}

    /// Renders the slider using ImGui with a dynamic minimum value.
    void render() override 
    {
    ImGui::SliderFloat(label.c_str(), value, *minValue, maxValue);
    }
private:
    std::string label; ///< The label of the slider.
    float* value;      ///< The float value that the slider modifies.
    float* minValue;   ///< A reference to the minimum value of the slider.
    float maxValue;    ///< The maximum value of the slider.
};

/**
 * @class SliderElementLowBoundedI
 * @brief A slider element for integers where the minimum value is dynamically set by a reference.
 * 
 * Similar to `SliderElementLowBounded`, but for integers.
 */
class SliderElementLowBoundedI: public Element
{
public:
    /**
     * @brief Constructs a SliderElementLowBounded with the specified parameters.
     * @param label The label to display next to the slider.
     * @param value A reference to the int value that the slider will modify.
     * @param minValue A reference to the minimum value of the slider.
     * @param maxValue The maximum value of the slider.
     */
    SliderElementLowBoundedI(std::string label, int& value, int& minValue, int maxValue)
        : label(label), value(&value), minValue(&minValue), maxValue(maxValue) {}

    /// Renders the slider using ImGui with a dynamic minimum value.
    void render() override 
    {
    ImGui::SliderInt(label.c_str(), value, *minValue, maxValue);
    }
private:
    std::string label; ///< The label of the slider.
    int* value;      ///< The int value that the slider modifies.
    int* minValue;   ///< A reference to the minimum value of the slider.
    int maxValue;    ///< The maximum value of the slider.
};

/**
 * @class CheckboxElement
 * @brief A UI element representing a checkbox.
 * 
 * This element renders a checkbox that can be checked or unchecked.
 */
class CheckboxElement: public Element
{
public:
    /**
     * @brief Constructs a CheckboxElement with the specified label and value.
     * @param label The label to display next to the checkbox.
     * @param value A reference to the boolean value that the checkbox will modify.
     */
    CheckboxElement(std::string label, bool& value)
        : label(label), value(&value) {}
    
    /**
     * @brief Constructs a CheckboxElement with a pointer to the value.
     * @param label The label to display next to the checkbox.
     * @param value A pointer to the boolean value that the checkbox will modify.
     */
    CheckboxElement(std::string label, bool* value)
        : label(label), value(value) {}

    /// Renders the checkbox using ImGui.
    void render() override 
    {
        ImGui::Checkbox(label.c_str(), value);
    }
private:
    std::string label; ///< The label of the checkbox.
    bool* value;       ///< The boolean value that the checkbox modifies.

};

/**
 * @class ButtonElement
 * @brief A UI element representing a button.
 * 
 * This element renders a button that triggers an action when clicked.
 */
class ButtonElement: public Element
{
public:
    /**
     * @brief Constructs a ButtonElement with the specified label and action.
     * @param label The label to display on the button.
     * @param action A function to execute when the button is pressed.
     */
    ButtonElement(std::string label, std::function<void()> action)
        : label(label), m_action(action) {}

    /// Renders the button using ImGui and executes the action when clicked.
    void render() override 
    {
        if (ImGui::Button(label.c_str())) {
            m_action();  // Execute action when pressed
        }
    }
private:
    std::string label; ///< The label of the button.
    std::function<void()> m_action; ///< The function to execute when the button is clicked.
};

/**
 * @class TextElementi
 * @brief A UI element displaying an integer value as text.
 * 
 * This element renders a label and its associated integer value as text.
 */
class TextElementi : public Element 
{
private:
    std::string label;  ///< The label of the text element.
    int* value;         ///< Pointer to the integer value to display.

public:
    /**
     * @brief Constructs a TextElementi with the specified label and integer value.
     * @param label string: The label to display.
     * @param value int&: A reference to the integer value to display.
     */
    TextElementi(std::string label, int& value)
        : label(label), value(&value) {}
    
    /**
     * @brief Constructs a TextElementi with the specified label and a pointer to an integer value.
     * @param label The label to display.
     * @param value A pointer to the integer value to display.
     */
    TextElementi(std::string label, int* value)
    : label(label), value(value) {}

    /// Renders the text label and value using ImGui
    void render() override 
    {
        ImGui::Text("%s: %d", label.c_str(), *value);
    }
};

/**
 * @class TextElement
 * @brief A UI element displaying a static text label.
 * 
 * This element renders a simple text label without any associated value.
 */
class TextElement : public Element 
{
private:

    std::string label;  ///< The label of the text element.

public:
    /**
     * @brief Constructs a TextElement with the specified label.
     * @param label The label to display.
     */
    TextElement(std::string label)
        : label(label) {}

    /// Renders the static text label using ImGui.
    void render() override 
    {
        ImGui::Text("%s", label.c_str());
    }
};

/**
 * @class TextElementf
 * @brief A UI element displaying a float value as text.
 * 
 * This element renders a label and its associated float value as text.
 */
class TextElementf : public Element 
{
private:
    std::string label;  ///< The label of the text element.
    float* value;       ///< Pointer to the float value to display.

public:
    /**
     * @brief Constructs a TextElementf with the specified label and float value.
     * @param label The label to display.
     * @param value A reference to the float value to display.
     */
    TextElementf(std::string label, float& value)
        : label(label), value(&value) {}
    
    /**
     * @brief Constructs a TextElementf with the specified label and a pointer to a float value.
     * @param label The label to display.
     * @param value A pointer to the float value to display.
     */
    TextElementf(std::string label, float* value)
    : label(label), value(value) {}

    /// Renders the text label and value using ImGui.
    void render() override 
    {
        ImGui::Text("%s: %.f", label.c_str(), *value);
    }
};

/**
 * @class Vec2TextElement
 * @brief A UI element displaying a glm::vec2 value as text.
 * 
 * This element renders a label and its associated glm::vec2 value as text.
 */
class Vec2TextElement: public Element
{
private:
    std::string label;  ///< The label of the text element.
    glm::vec2* value;   ///< Pointer to the glm::vec2 value to display.

public:
    /**
     * @brief Constructs a Vec2TextElement with the specified label and glm::vec2 value.
     * @param label The label to display.
     * @param value A reference to the glm::vec2 value to display.
     */
    Vec2TextElement(std::string label, glm::vec2& value)
        : label(label), value(&value) {}

    /// Renders the text label and glm::vec2 value using ImGui.
    void render() override 
    {
        ImGui::Text("%s: (%.2f, %.2f)", label.c_str(), value->x, value->y);
    }
};

/**
 * @class Vec3TextElement
 * @brief A UI element displaying a glm::vec3 value as text.
 * 
 * This element renders a label and its associated glm::vec3 value as text.
 */
class Vec3TextElement: public Element
{
private:
    std::string label;  ///< The label of the text element.
    glm::vec3* value;   ///< Pointer to the glm::vec3 value to display.

public:
    /**
     * @brief Constructs a Vec3TextElement with the specified label and glm::vec3 value.
     * @param label The label to display.
     * @param value A reference to the glm::vec3 value to display.
     */
    Vec3TextElement(std::string label, glm::vec3& value)
        : label(label), value(&value) {}

    /// Renders the text label and glm::vec3 value using ImGui.
    void render() override 
    {
        ImGui::Text("%s: (%.2f, %.2f, %.2f)", label.c_str(), value->x, value->y, value->z);
    }
};

/**
 * @class SameLineElement
 * @brief A UI element that renders content on the same line.
 * 
 * This element renders a space and allows other elements to be placed on the same line.
 */
class SameLineElement: public Element
{
    /// Renders the same line behavior using ImGui.
    void render() override
    {
        ImGui::SameLine();
        ImGui::Text(" ");
        ImGui::SameLine();
    }
};

/**
 * @class NewLineElement
 * @brief A UI element that forces a new line.
 * 
 * This element renders a new line in the UI.
 */
class NewLineElement: public Element
{
    /// Renders the new line behavior using ImGui.
    void render() override
    {
        ImGui::NewLine();
    }
};

#endif // _ELEMENT_H_