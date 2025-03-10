#ifndef _APP_UI_H_
#define _APP_UI_H_

#include <imgui.h>
#include <memory>
#include <unordered_map>
#include <string>
#include "ui/component.h"
#include "ui/components/scene_info.h"
#include "ui/components/camera_info.h"
#include "ui/components/input_info.h"
#include "ui/components/benchmark_info.h"

// #include "ux/cinematic/benchmark.h"

class AppUI
{
public:
    void addSceneInfoComponent(std::string name, std::unordered_map<std::string, int*> scene_data);
    void addCameraInfoComponent(std::string name, Camera* camera);
    void addInputInfoComponent(std::string name, Input* input);
    void addBenchmarkInfoComponent(std::string name, Benchmark* benchmark);

    void render();
    
private:
    std::vector<std::unique_ptr<Component>> components;
};

#endif