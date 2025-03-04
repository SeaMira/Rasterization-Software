#ifndef CAMERA_CONTROLLED_PATH_H
#define CAMERA_CONTROLLED_PATH_H

#include <vector>
#include "utils/math_ops.h"
#include "ux/camera.h"
#include "ux/input.h"

class CameraControlledPath 
{
public:
    struct Checkpoint 
    {
        glm::vec3 position;
        glm::vec3 target;

        Checkpoint(glm::vec3 pos, glm::vec3 tgt) : position(pos), target(tgt) {}
    };

    CameraControlledPath(Camera& camera, Input& input) : 
        m_camera(&camera), m_input(&input), m_currentIndex(0), m_progress(0.0f), m_speed(1.0f) {}

    void addCheckpoint(const glm::vec3& pos, const glm::vec3& tgt);
    void reset();
    void update(); 
    void nextCheckpoint();
    void lastCheckpoint();

    glm::vec3 getPosition() const;
    glm::vec3 getTarget() const;

    void setSpeed(float newSpeed);
    float getSpeed() const;


private:
    Camera* m_camera;
    Input* m_input;

    std::vector<Checkpoint> m_checkpoints;
    int m_currentIndex;
    float m_progress;
    float m_speed;
};

#endif // CAMERA_CONTROLLED_PATH_H