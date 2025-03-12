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