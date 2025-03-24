#ifndef _BENCHMARK_INFO_H_
#define _BENCHMARK_INFO_H_

#include <vector>
// #include "ux/cinematic/benchmark.h"
#include "ui/component.h"

class Benchmark;

/**
 * @class BenchmarkInfoComponent
 * @brief Represents a UI component for displaying benchmark information.
 * 
 * This class provides functionality for rendering a UI component that displays information about the current benchmark, for example, frame, if its playing and if the 
 * user is currently doing the benchmark.
 */
class BenchmarkInfoComponent : public Component
{
public:
    /**
     * @brief Constructs a BenchmarkInfoComponent with the specified name and benchmark object.
     * 
     * Takes a Benchmark object and a name for the component and initializes the component with the UI elements it will show on the
     * interface.
     */
    BenchmarkInfoComponent(std::string name, Benchmark* benchmark);
    
    /**
     * @brief Moves to the previous checkpoint in the benchmark.
     * 
     * Utilizes the Benchmark object methods to move to the previous checkpoint in the benchmark. Is called when the user clicks the "Prev" UI button.
     */
    void prevCheckpoint();

    /**
     * @brief Moves to the next checkpoint in the benchmark.
     * 
     * Utilizes the Benchmark object methods to move to the next checkpoint in the benchmark. Is called when the user clicks the "Next" UI button.
     */
    void nextCheckpoint();

    /**
     * @brief Renders the BenchmarkInfoComponent.
     * 
     * Renders the BenchmarkInfoComponent by rendering all its associated UI elements. If the "On Benchmark" checkbox is checked, it will render all the elements, otherwise
     * it will only render the checkbox.
     */
    void render() override;
private:
    Benchmark* m_benchmark; ///< The Benchmark object associated with the component.
    bool* m_on_benchmark; ///< A pointer to a boolean that indicates if the user is currently doing the benchmark.
};


#endif // _BENCHMARK_INFO_H_