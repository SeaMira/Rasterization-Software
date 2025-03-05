#include "ux/cinematic/benchmark.h"

Benchmark::Benchmark(CameraController& camera_controller, std::vector<std::pair<glm::vec3, glm::vec3>> checkpoints):
m_camera_controller(&camera_controller), m_camera_path(*(camera_controller.cameraHandle()), *(camera_controller.inputHandle()))
{
    m_camera_controller->setBenchmark(true);
    for (auto& checkpoint: checkpoints)
    {
        m_camera_path.addCheckpoint(checkpoint.first, checkpoint.second);
    }
}
        
void Benchmark::goToNextCheckpoint()
{
    m_camera_path.nextCheckpoint();
}

void Benchmark::goToLastCheckpoint()
{
    m_camera_path.lastCheckpoint();

}

void Benchmark::speedUp()
{
    m_camera_path.setSpeed(m_camera_path.getSpeed() + m_camera_controller->inputHandle()->deltaTime);
}

void Benchmark::play()
{
    m_play = true;
}

void Benchmark::stop()
{
    m_play = false;
}

void Benchmark::speedDown()
{
    m_camera_path.setSpeed(std::max(0.0f, m_camera_path.getSpeed() - m_camera_controller->inputHandle()->deltaTime));

}

void Benchmark::addCheckpoint(const glm::vec3& pos, const glm::vec3& tgt)
{
    m_camera_path.addCheckpoint(pos, tgt);
}

void Benchmark::reset()
{
    m_camera_path.reset();
}

void Benchmark::update()
{
    if (m_camera_controller->onBenchmark())
    {
        checkInput();
        if (m_play) m_camera_path.update();
    }
}

void Benchmark::checkInput()
{
    if (m_camera_controller->inputHandle()->isKeyDown(Key::P))
    {
        if (m_play) stop();
        else play();
    }
    
    if (m_camera_controller->inputHandle()->isKeyDown(Key::R)) reset();
    if (m_camera_controller->inputHandle()->isKeyDown(Key::Right)) goToNextCheckpoint();
    if (m_camera_controller->inputHandle()->isKeyDown(Key::Left)) goToLastCheckpoint();
    if (m_camera_controller->inputHandle()->isKeyDown(Key::Up)) speedUp();
    if (m_camera_controller->inputHandle()->isKeyDown(Key::Down)) speedDown();

}

int Benchmark::getCheckpointID() const
{
    return m_camera_path.getCurrentIndex();
}