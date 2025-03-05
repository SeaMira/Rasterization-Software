#ifndef _BENCHMARK_H_
#define _BENCHMARK_H_

#include <vector>
#include "ux/cinematic/camera_controlled_movement.h"
#include "ux/camera_controller.h"


class Benchmark
{
    public:
        Benchmark(CameraController& camera_controller, std::vector<std::pair<glm::vec3, glm::vec3>> checkpoints);
        
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

        int getCheckpointID() const;
    private:
        CameraController* m_camera_controller;
        CameraControlledPath m_camera_path;
        bool m_play = true;

};

#endif // _BENCHMARK_H_