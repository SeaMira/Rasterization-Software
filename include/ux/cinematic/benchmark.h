#ifndef _BENCHMARK_H_
#define _BENCHMARK_H_

#include <vector>
#include "ux/cinematic/camera_controlled_movement.h"
#include "ux/input.h"
#include "ux/camera.h"


class Benchmark
{
    public:
        Benchmark(Camera& camera, Input& input, std::vector<std::pair<glm::vec3, glm::vec3>> checkpoints);
        
        void goToNextCheckpoint();
        void goToLastCheckpoint();
        void speedUp();
        void play();
        void stop();
        void speedDown();

        void addCheckpoint(const glm::vec3& pos, const glm::vec3& tgt);
        void reset();
        
        void update();
        void checkInput();
    private:
        Camera* m_camera;
        Input* m_input;
        CameraControlledPath m_camera_path;
        bool m_play = true;

};

#endif // _BENCHMARK_H_