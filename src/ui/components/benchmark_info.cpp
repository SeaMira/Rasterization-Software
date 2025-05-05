#include "ui/components/benchmark_info.h"
#include "ux/cinematic/benchmark.h"

void BenchmarkInfoComponent::prevCheckpoint()
{
    m_benchmark->goToLastCheckpoint();
}

void BenchmarkInfoComponent::nextCheckpoint()
{
    m_benchmark->goToNextCheckpoint();
}

BenchmarkInfoComponent::BenchmarkInfoComponent(std::string name, Benchmark* benchmark) :
    Component(name), m_benchmark(benchmark)
{
    m_on_benchmark = m_benchmark->m_camera_controller->onBenchmarkData();

    addElement(std::make_unique<CheckboxElement>("On Benchmark", m_on_benchmark));
    addElement(std::make_unique<CheckboxElement>("Playing", &m_benchmark->m_play));
    addElement(std::make_unique<ButtonElement>("Prev",
        [this]() { prevCheckpoint(); }
    ));
    addElement(std::make_unique<SameLineElement>());
    addElement(std::make_unique<TextElementi>("Frame", m_benchmark->m_camera_path.getCurrentIndexData()));
    addElement(std::make_unique<SameLineElement>());
    addElement(std::make_unique<ButtonElement>("Next",
        [this]() { nextCheckpoint(); }
    ));
    addElement(std::make_unique<SliderElementTopBoundedI>(
        "Initial track checkpoint", m_benchmark->m_initial_checkpoint, 0, m_benchmark->m_final_checkpoint
    ));
    addElement(std::make_unique<SliderElementLowBoundedI>(
        "Final track checkpoint", m_benchmark->m_final_checkpoint, m_benchmark->m_initial_checkpoint, m_benchmark->m_last_checkpoint_id - 1
    ));
    addElement(std::make_unique<ButtonElement>("Play Track",
        [this]() { m_benchmark->playTrack(); }
    ));
}

void BenchmarkInfoComponent::render()
{
     elements[0]->render();  

    // Si el checkbox está marcado, mostramos el resto de elementos
    if (*m_on_benchmark) {
        for (size_t i = 1; i < elements.size(); i++) {
            elements[i]->render();
        }
    }
}