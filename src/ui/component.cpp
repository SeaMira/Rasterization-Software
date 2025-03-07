#include "ui/component.h"

void Component::render()
{
    for (auto& element : elements)
    {
        element->render();
    }
}