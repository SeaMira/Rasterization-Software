#ifndef _ELEMENT_H_
#define _ELEMENT_H_
#include <imgui.h>
#include <glm/glm.hpp>
#include <string>
#include <functional> 

class Element {
public:
    virtual ~Element() = default;

    virtual void render() = 0;
};


class SliderElement: public Element
{
    
public:
    // Constructor que recibe los parámetros del slider
    SliderElement(std::string label, float& value, float minValue, float maxValue)
        : label(label), value(&value), minValue(minValue), maxValue(maxValue) {}

    // Implementación del método render
    void render() override 
    {
    ImGui::SliderFloat(label.c_str(), value, minValue, maxValue);
    }
private:
    std::string label;
    float* value;
    float minValue;
    float maxValue;
};

class SliderElementTopBounded: public Element
{
    
public:
    // Constructor que recibe los parámetros del slider
    SliderElementTopBounded(std::string label, float& value, float minValue, float& maxValue)
        : label(label), value(&value), minValue(minValue), maxValue(&maxValue) {}

    // Implementación del método render
    void render() override 
    {
    ImGui::SliderFloat(label.c_str(), value, minValue, *maxValue);
    }
private:
    std::string label;
    float* value;
    float minValue;
    float* maxValue;
};

class SliderElementLowBounded: public Element
{
    
public:
    // Constructor que recibe los parámetros del slider
    SliderElementLowBounded(std::string label, float& value, float& minValue, float maxValue)
        : label(label), value(&value), minValue(&minValue), maxValue(maxValue) {}

    // Implementación del método render
    void render() override 
    {
    ImGui::SliderFloat(label.c_str(), value, *minValue, maxValue);
    }
private:
    std::string label;
    float* value;
    float* minValue;
    float maxValue;
};

class CheckboxElement: public Element
{
public:
    // Constructor que recibe el valor del checkbox
    CheckboxElement(std::string label, bool& value)
        : label(label), value(&value) {}
    
    CheckboxElement(std::string label, bool* value)
        : label(label), value(value) {}

    // Implementación del método render
    void render() override 
    {
        ImGui::Checkbox(label.c_str(), value);
    }
private:
    std::string label;
    bool* value;

};

class ButtonElement: public Element
{
public:
    // Constructor that recieves button label and associated action
    ButtonElement(std::string label, std::function<void()> action)
        : label(label), m_action(action) {}

    // Render method implementation
    void render() override 
    {
        if (ImGui::Button(label.c_str())) {
            m_action();  // Execute action when pressed
        }
    }
private:
    std::string label;
    std::function<void()> m_action;  // Función a ejecutar cuando se hace clic en el botón
};

class TextElementi : public Element 
{
private:
    std::string label;  // Etiqueta del texto
    int* value;         // Puntero al valor entero

public:
    // Constructor que recibe el label y una referencia al valor entero
    TextElementi(std::string label, int& value)
        : label(label), value(&value) {}
    
    TextElementi(std::string label, int* value)
    : label(label), value(value) {}

    // Implementación del método render
    void render() override 
    {
        ImGui::Text("%s: %d", label.c_str(), *value);
    }
};


class TextElement : public Element 
{
private:
    std::string label;  // Etiqueta del texto

public:
    // Constructor que recibe el label y una referencia al valor entero
    TextElement(std::string label)
        : label(label) {}

    // Implementación del método render
    void render() override 
    {
        ImGui::Text("%s", label.c_str());
    }
};

class TextElementf : public Element 
{
private:
    std::string label;  // Etiqueta del texto
    float* value;         // Puntero al valor entero

public:
    // Constructor que recibe el label y una referencia al valor entero
    TextElementf(std::string label, float& value)
        : label(label), value(&value) {}
    
    TextElementf(std::string label, float* value)
    : label(label), value(value) {}

    // Implementación del método render
    void render() override 
    {
        ImGui::Text("%s: %.f", label.c_str(), *value);
    }
};

class Vec2TextElement: public Element
{
private:
    std::string label;  // Etiqueta del vector
    glm::vec2* value;   // Puntero al vector 3D

public:
    // Constructor que recibe el label y una referencia al glm::vec3
    Vec2TextElement(std::string label, glm::vec2& value)
        : label(label), value(&value) {}

    // Implementación del método Render
    void render() override 
    {
        ImGui::Text("%s: (%.2f, %.2f)", label.c_str(), value->x, value->y);
    }
};

class Vec3TextElement: public Element
{
private:
    std::string label;  // Etiqueta del vector
    glm::vec3* value;   // Puntero al vector 3D

public:
    // Constructor que recibe el label y una referencia al glm::vec3
    Vec3TextElement(std::string label, glm::vec3& value)
        : label(label), value(&value) {}

    // Implementación del método Render
    void render() override 
    {
        ImGui::Text("%s: (%.2f, %.2f, %.2f)", label.c_str(), value->x, value->y, value->z);
    }
};

class SameLineElement: public Element
{
    void render() override
    {
        ImGui::SameLine();
        ImGui::Text(" ");
        ImGui::SameLine();
    }
};

class NewLineElement: public Element
{
    void render() override
    {
        ImGui::NewLine();
    }
};

#endif // _ELEMENT_H_