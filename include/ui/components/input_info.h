#ifndef _INPUT_INFO_H_
#define _INPUT_INFO_H_

#include <vector>
#include "ux/input.h"
#include "ui/component.h"

class InputInfoComponent : public Component
{
public:
    InputInfoComponent(std::string name, Input* input);

    void screenshotAction();
private:
    Input* m_input;
};


#endif // _INPUT_INFO_H_