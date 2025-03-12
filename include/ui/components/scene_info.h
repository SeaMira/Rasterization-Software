#ifndef _SCENE_INFO_H_
#define _SCENE_INFO_H_

#include <unordered_map>
#include "ui/component.h"
#include "ui/element.h"

class SceneInfoComponent : public Component
{
public:
    SceneInfoComponent(std::string name, std::unordered_map<std::string, int*> scene_data);

};


#endif // _SCENE_INFO_H_