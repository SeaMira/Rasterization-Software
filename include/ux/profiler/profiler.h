#ifndef _PROFILER_H_
#define _PROFILER_H_

#include <string>
#include <vector>
#include "vis/window.h"

class Profiler
{
public:
    Profiler(Window& window, std::string frame_times_file, 
        std::string process_times_file, double seconds_timer = 30.0);
    Profiler( const Profiler & )             = delete;
    Profiler & operator=( const Profiler & ) = delete;

    Profiler( Profiler && other ) noexcept;
    Profiler & operator=( Profiler && other ) noexcept;

    ~Profiler() = default;

    void startSavingNextFrames(int checkpoint);
    void updateProfiler();

    void startSavingNextProcessTime();
    void finishSavingNextProcessTime();

    void writeToFramesCounter_off(int checkpoint);
    void writeToProcessTime_off(std::string process_name);
private:
    Window* m_window;
    
    uint64_t m_frequency;

    int m_checkpoint;
    uint64_t m_frame_start_time;
    uint64_t m_frame_current_time;
    double m_last_frame_sec_dt;
    double m_seconds_timer;
    std::vector<double> m_dts;

    uint64_t m_process_start_time;
    uint64_t m_process_total_time;
    
    std::string m_frame_times_file;
    std::string m_process_times_file;

    bool checking_frames = false;
};

#endif // _PROFILER_H_