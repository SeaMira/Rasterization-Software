#ifndef _COMPONENT_H_
#define _COMPONENT_H_

#include <imgui.h>
#include <vector>
#include <string>
#include <memory>
#include "ui/element.h"

class Component {
public:
    Component(std::string name) : m_name(name) {}
    virtual ~Component() = default;

    // Component rendering
    virtual void render();

    inline void addElement(std::unique_ptr<Element> e) { elements.push_back(std::move(e));}
    // Menu name getter
    inline std::string& getName() { return m_name; }

protected:
    std::string m_name;
    std::vector<std::unique_ptr<Element>> elements;
};

#endif // _COMPONENT_H_