#include "ux/cinematic/benchmark.h"

Benchmark::Benchmark(Camera& camera, Input& input, std::vector<std::pair<glm::vec3, glm::vec3>> checkpoints):
m_camera(&camera), m_input(&input), m_camera_path(camera, input)
{
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
    m_camera_path.setSpeed(m_camera_path.getSpeed() + m_input->deltaTime);
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
    m_camera_path.setSpeed(std::min(0.0f, m_camera_path.getSpeed() - m_input->deltaTime));

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
    checkInput();
    if (m_play) m_camera_path.update();
}

void Benchmark::checkInput()
{
    if (m_input->isKeyDown(Key::P))
    {
        if (m_play) stop();
        else play();
    }
    
    if (m_input->isKeyDown(Key::R)) reset();
    if (m_input->isKeyDown(Key::Right)) goToNextCheckpoint();
    if (m_input->isKeyDown(Key::Left)) goToLastCheckpoint();
    if (m_input->isKeyDown(Key::Up)) speedUp();
    if (m_input->isKeyDown(Key::Down)) speedDown();

}