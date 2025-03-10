#ifndef _BENCHMARK_INFO_H_
#define _BENCHMARK_INFO_H_

#include <vector>
// #include "ux/cinematic/benchmark.h"
#include "ui/component.h"

class Benchmark;

class BenchmarkInfoComponent : public Component
{
public:
    BenchmarkInfoComponent(std::string name, Benchmark* benchmark);
    void prevCheckpoint();
    void nextCheckpoint();
    void render() override;
private:
    Benchmark* m_benchmark;
    bool* m_on_benchmark;
};


#endif // _BENCHMARK_INFO_H_