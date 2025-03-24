#ifndef _PROFILER_H_
#define _PROFILER_H_

#include <string>
#include <vector>
#include "vis/window.h"

/**
 * @class Profiler
 * 
 * @brief Profiler class for saving frame times and process times .
 * 
 * The Profiler class is used to save frame times and process times to a file for later analysis.
 */
class Profiler
{
public:
    /**
     * @brief Constructor for the Profiler class.
     * 
     * Constructor for the Profiler class, initializes the profiler with the window and the files to save the times.
     * 
     * @param window the window to be used for the profiler.
     * @param frame_times_file the file to save the frame times.
     * @param process_times_file the file to save the process times.
     * @param seconds_timer the time the saving of frame times takes.
     */
    Profiler(Window& window, std::string frame_times_file, 
        std::string process_times_file, double seconds_timer = 30.0);

    /**
     * @brief Copy constructor for the Profiler class.
     */
    Profiler( const Profiler & )             = delete;

    /**
     * @brief Copy assignment operator for the Profiler class.
     */
    Profiler & operator=( const Profiler & ) = delete;

    /**
     * @brief Move constructor for the Profiler class.
     */
    Profiler( Profiler && other ) noexcept;

    /**
     * @brief Move assignment operator for the Profiler class.
     */
    Profiler & operator=( Profiler && other ) noexcept;

    /**
     * @brief Destructor for the Profiler class.
     * 
     * Default destructor for the Profiler class. No need for manual deallocation.
     */
    ~Profiler() = default;

    /**
     * @brief start saving the next frames.
     * 
     * Starts saving the next frames to the file.
     * 
     * @param checkpoint the checkpoint it will start saving the frames from.
     */
    void startSavingNextFrames(int checkpoint);

    /**
     * @brief updates the frames saving.
     * 
     * If the frames are being saved, it updates the frame saving process. If the saving time
     * passed it writes on the file of frame times.
     */
    void updateProfiler();

    /**
     * @brief starts chronometer and flags for saving the next process time.
     */
    void startSavingNextProcessTime();

    /**
     * @brief finishes the saving of the next process time.
     * 
     * Finishes the saving of the next process time and calculates the process duration.
     */
    void finishSavingNextProcessTime();

    /**
     * @brief writes the frame times to the file.
     * 
     * Writes the frame times to the file for a certain checkpoint.
     */
    void writeToFramesCounter_off(int checkpoint);

    /**
     * @brief writes the process time to the file.
     * 
     * Writes the process time to the file for a certain named process.
     */
    void writeToProcessTime_off(std::string process_name);
private:
    Window* m_window; ///< the window to be used for the profiler.
    
    uint64_t m_frequency; ///< the frequency of the performance counter.

    int m_checkpoint; ///< the checkpoint to start saving the frames from.
    uint64_t m_frame_start_time; ///< the start time of saving frames.
    uint64_t m_frame_current_time; ///< the current time of the frame time saving.
    double m_last_frame_sec_dt; ///< the last frame time in seconds.
    double m_seconds_timer; ///< the time spent saving the frame times.
    std::vector<double> m_dts; ///< the frame times to be saved.

    uint64_t m_process_start_time; ///< the start time of the process time saving.
    uint64_t m_process_total_time; ///< the total time of the process time saving.
    
    std::string m_frame_times_file; ///< the file to save the frame times.
    std::string m_process_times_file; ///< the file to save the process times.

    bool checking_frames = false; ///< flag for checking if the frames are being saved.
};

#endif // _PROFILER_H_