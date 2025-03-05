#include "ux/cinematic/camera_controlled_movement.h"


void CameraControlledPath::addCheckpoint(const glm::vec3& pos, const glm::vec3& tgt) 
{
    m_checkpoints.emplace_back(pos, tgt);
}

void CameraControlledPath::reset() 
{
    m_progress = 0.0f;
    m_currentIndex = 0;
}

void CameraControlledPath::update() 
{
    if (m_checkpoints.size() < 2) return;

    m_progress += m_input->deltaTime * m_speed;
    if (m_progress > 1.0f) 
    {
        m_progress = 0.0f;
        m_currentIndex++;
        if (m_currentIndex >= m_checkpoints.size() - 1) 
        {
            m_currentIndex = 0; // Reiniciar ciclo para benchmarks
        }
    }

    // Obtener los checkpoints actuales
    Checkpoint& start = m_checkpoints[m_currentIndex];
    Checkpoint& end = m_checkpoints[m_currentIndex + 1];

    // Interpolación de posición (LERP)
    m_camera->SetPosition(glm::mix(start.position, end.position, m_progress));
    m_camera->lookAtTarget(glm::mix(start.target, end.target, m_progress));

}

void CameraControlledPath::nextCheckpoint()
{
    if (m_checkpoints.size() < 2) return;
    m_progress = 0.0f;
    if (m_currentIndex < m_checkpoints.size() - 1) 
    {
        m_currentIndex++;
        Checkpoint& pointToCheck = m_checkpoints[m_currentIndex];
        m_camera->SetPosition(pointToCheck.position);
        m_camera->lookAtTarget(pointToCheck.target);
    }
}

void CameraControlledPath::lastCheckpoint()
{
    if (m_checkpoints.size() < 2) return;
    m_progress = 0.0f;
    if (m_currentIndex > 0) 
    {
        m_currentIndex--;
        Checkpoint& pointToCheck = m_checkpoints[m_currentIndex];
        m_camera->SetPosition(pointToCheck.position);
        m_camera->lookAtTarget(pointToCheck.target);
    }
}


glm::vec3 CameraControlledPath::getPosition() const { return m_camera->getPosition(); }
glm::vec3 CameraControlledPath::getTarget() const { return m_checkpoints[m_currentIndex].target; }

void CameraControlledPath::setSpeed(float newSpeed) { m_speed = newSpeed; }
float CameraControlledPath::getSpeed() const { return m_speed; }

int CameraControlledPath::getCurrentIndex() const { return m_currentIndex; }