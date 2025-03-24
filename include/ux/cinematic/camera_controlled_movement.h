#ifndef CAMERA_CONTROLLED_PATH_H
#define CAMERA_CONTROLLED_PATH_H

#include <vector>
#include "utils/math_ops.h"
#include "ux/camera.h"
#include "ux/input.h"

/**
 * @class CameraControlledPath
 * 
 * @brief CameraControlledPath class for controlling the camera along a path looking at
 * certain targets.
 */
class CameraControlledPath 
{
public:

    /**
     * @struct Checkpoint
     * 
     * @brief Checkpoint struct for storing the position and target of a checkpoint.
     */
    struct Checkpoint 
    {
        glm::vec3 position; ///< the position of the checkpoint.
        glm::vec3 target; ///< the target to look at of the checkpoint.

        Checkpoint(glm::vec3 pos, glm::vec3 tgt) : position(pos), target(tgt) {}
    };

    /**
     * @brief Constructor for the CameraControlledPath class.
     * 
     * Constructor for the CameraControlledPath class, initializes the camera controlled path with the camera
     * and the input class that controls it and starting all values on neutral values.
     */
    CameraControlledPath(Camera& camera, const Input& input) : 
        m_camera(&camera), m_input(&input), m_currentIndex(0), m_progress(0.0f), m_speed(1.0f) {}

    /**
     * @brief add a checkpoint to the path and its target.
     * 
     * Adds a checkpoint to the path and its target so it will now have an interpolation, and thus the camera can walk through it.
     * 
     * @param pos the position of the checkpoint.
     * @param tgt the target of the checkpoint.
     */
    void addCheckpoint(const glm::vec3& pos, const glm::vec3& tgt);

    /**
     * @brief reset the path.
     * 
     * Resets the camera to the first checkpoint.
     */
    void reset();

    /**
     * @brief update the path.
     * 
     * Updates the camera position and target so it moves along the path the corresponding frame.
     */
    void update(); 

    /**
     * @brief go to the next checkpoint.
     * 
     * Sets the camera to the next checkpoint in the path.
     */
    void nextCheckpoint();

    /**
     * @brief go to the last checkpoint.
     * 
     * Sets the camera to the last checkpoint in the path.
     */
    void lastCheckpoint();

    /**
     * @brief gets camera position.
     * 
     * Gets the current position of the camera.
     */
    glm::vec3 getPosition() const;

    /**
     * @brief get the current target.
     * 
     * Gets the current target of the camera.
     * 
     * @return the current target of the camera.
     */
    glm::vec3 getTarget() const;

    /**
     * @brief set camera speed.
     * 
     * @param newSpeed the new speed of the camera.
     */
    void setSpeed(float newSpeed);

    /**
     * @brief get the current camera speed.
     * 
     * @return the current speed of the camera.
     */
    float getSpeed() const;

    /**
     * @brief get the current checkpoint index.
     * 
     * @return the current checkpoint index of the path.
     */
    int getCurrentIndex() const;

    /**
     * @brief get the current index location data.
     * 
     * Gets the index location data for accessing from different classes
     * 
     * @return the current index location data checkpoints.
     */
    int* getCurrentIndexData() {return &m_currentIndex; }

private:
    Camera* m_camera; ///< the camera to be controlled.
    const Input* m_input; ///< the input to be used for the camera.

    std::vector<Checkpoint> m_checkpoints; ///< the checkpoints of the path.
    int m_currentIndex; ///< the current checkpoint index of the path.
    float m_progress; ///< the progress of the camera along the path of two consecutive checkpoints.
    float m_speed; ///< the speed of the camera along the path.
};

#endif // CAMERA_CONTROLLED_PATH_H