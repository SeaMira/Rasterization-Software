#ifndef _BENCHMARK_H_
#define _BENCHMARK_H_

#include <vector>
#include "ux/cinematic/camera_controlled_movement.h"
#include "ux/camera_controller.h"
// #include "ui/components/benchmark_info.h"

/**
 * @class Benchmark
 * 
 * @brief Benchmark class for controlling the camera along a path.
 * 
 * The Benchmark class is used to control the camera along a path of checkpoints and targets at runtime.
 * Keeps on check the movement between checkpoints and the speed of the camera, and allows getting info
 * of it from the main application.
 */
class Benchmark
{
    public:
        /**
         * @brief Constructor for the Benchmark class.
         * 
         * Constructor for the Benchmark class, initializes the camera controlled path with the checkpoints.
         * 
         * @param camera_controller the camera controller to be used for the benchmark.
         * @param checkpoints the checkpoints to be used for the benchmark.
         */
        Benchmark(CameraController& camera_controller, std::vector<std::pair<glm::vec3, glm::vec3>> checkpoints);
        
        /**
         * @brief go to the next checkpoint.
         * 
         * Moves the camera to the next checkpoint in the path.
         */
        void goToNextCheckpoint();

        /**
         * @brief go to the last checkpoint.
         * 
         * Moves the camera to the last checkpoint in the path.
         */
        void goToLastCheckpoint();

        /**
         * @brief speed up the camera.
         * 
         * Speeds the camera up along the path.
         */
        void speedUp();

        /**
         * @brief play the benchmark.
         * 
         * Plays the benchmark so every frame contributes to camera movement.
         */
        void play();

        /**
         * @brief stop the benchmark.
         * 
         * Stops the benchmark so the camera does not move.
         */
        void stop();

        /**
         * @brief speed the camera down.
         */
        void speedDown();

        /**
         * @brief add a checkpoint to the path and its target.
         * 
         * Adds a checkpoint to the path and its target so it will now have an interpolation.
         * 
         * @param pos the position of the checkpoint.
         * @param tgt the target of the checkpoint.
         */
        void addCheckpoint(const glm::vec3& pos, const glm::vec3& tgt);

        /**
         * @brief reset the benchmark.
         * 
         * Resets the benchmark so the camera goes back to the first checkpoint and default values.
         */
        void reset();
        
        /**
         * @brief update the benchmark.
         * 
         * Updates the benchmark so the camera moves along the path the corresponding frame.
         */
        void update();

        /**
         * @brief check the input for the benchmark.
         * 
         * Checks the input for the benchmark so the camera can be controlled.
         */
        void checkInput();

        /**
         * @brief get the current checkpoint ID.
         * 
         * Gets the current checkpoint ID so it can be used for the benchmark info by knowing which frame
         * is being traversed.
         * 
         * @return the current checkpoint ID.
         */
        int getCheckpointID() const;
    private:
        CameraController* m_camera_controller; ///< the camera controller to be used for the benchmark.
        CameraControlledPath m_camera_path; ///< the camera controlled path for the benchmark.
        bool m_play = true; ///< whether the benchmark is playing or not.

        friend class BenchmarkInfoComponent; ///< the benchmark info component is a friend class so it can be shown on UI.

};

#endif // _BENCHMARK_H_