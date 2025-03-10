#include "ui/components/input_info.h"

void InputInfoComponent::screenshotAction() {
    m_input->keysDown.emplace(Key::F10);
}

InputInfoComponent::InputInfoComponent(std::string name, Input* input) : 
    Component(name), m_input(input) 
{
    addElement(std::make_unique<TextElement>("Front/W - Back/S - Left/A - Right/D"));
    addElement(std::make_unique<TextElement>("Up/SPACE - Down/LShift"));
    addElement(std::make_unique<NewLineElement>());
    addElement(std::make_unique<TextElement>("While Right Click pressed:\nMove mouse = camera rotation.\nMouse wheel = Zoom in/out."));
    addElement(std::make_unique<NewLineElement>());
    addElement(std::make_unique<TextElement>("E = camera speed+\nQ = Camera speed-"));
    addElement(std::make_unique<NewLineElement>());
    addElement(std::make_unique<ButtonElement>("Screenshot (F10)", 
        [this]() { screenshotAction(); }
    ));


}

