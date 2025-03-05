#include <SDL3/SDL.h>
#include <iostream>
#include <fstream>
#include "ux/profiler/profiler.h"

Profiler::Profiler(Window& window, std::string frame_times_file, 
    std::string process_times_file, double seconds_timer) : 
    m_window(&window), m_seconds_timer(seconds_timer),
    m_frame_times_file(frame_times_file), m_process_times_file(process_times_file) 
{
    m_frequency = SDL_GetPerformanceFrequency();
}

Profiler::Profiler( Profiler && other ) noexcept
{
    std::swap( m_window, other.m_window );
    std::swap( m_frequency, other.m_frequency );
    std::swap( m_seconds_timer, other.m_seconds_timer );
    std::swap( m_dts, other.m_dts );
    std::swap( m_frame_times_file, other.m_frame_times_file );
    std::swap( m_process_times_file, other.m_process_times_file );
    
    std::swap( m_frame_start_time, other.m_frame_start_time );
    std::swap( m_frame_current_time, other.m_frame_current_time );
    std::swap( m_process_start_time, other.m_process_start_time );
    std::swap( m_process_total_time, other.m_process_total_time );
    std::swap( checking_frames, other.checking_frames );
}

Profiler & Profiler::operator=( Profiler && other ) noexcept
{
    std::swap( m_window, other.m_window );
    std::swap( m_frequency, other.m_frequency );
    std::swap( m_seconds_timer, other.m_seconds_timer );
    std::swap( m_dts, other.m_dts );
    std::swap( m_frame_times_file, other.m_frame_times_file );
    std::swap( m_process_times_file, other.m_process_times_file );

    std::swap( m_frame_start_time, other.m_frame_start_time );
    std::swap( m_frame_current_time, other.m_frame_current_time );
    std::swap( m_process_start_time, other.m_process_start_time );
    std::swap( m_process_total_time, other.m_process_total_time );
    std::swap( checking_frames, other.checking_frames );
    
    return *this;
}


void Profiler::startSavingNextFrames(int checkpoint)
{
    checking_frames = true;
    m_checkpoint = checkpoint;
    m_last_frame_sec_dt = 0.0;
    m_frame_start_time = SDL_GetPerformanceCounter();
    std::cout << "Started frame saving." << std::endl;
}

void Profiler::updateProfiler()
{
    if (checking_frames)
    {
        m_frame_current_time = SDL_GetPerformanceCounter();
        double elapsedSeconds = (double)(m_frame_current_time - m_frame_start_time) / m_frequency;
        m_dts.push_back(elapsedSeconds - m_last_frame_sec_dt);
        m_last_frame_sec_dt = elapsedSeconds;
        if (elapsedSeconds >= m_seconds_timer) 
        {
            writeToFramesCounter_off(m_checkpoint);
            checking_frames = false;
            std::vector<double>().swap(m_dts);
            std::cout << "Finished frame saving on " << elapsedSeconds << " seconds." << std::endl;
        }
    }
}

void Profiler::startSavingNextProcessTime()
{
    m_process_start_time = SDL_GetPerformanceCounter();
}

void Profiler::finishSavingNextProcessTime()
{
    m_process_total_time = (SDL_GetPerformanceCounter() - m_process_start_time)/m_frequency;
}

void Profiler::writeToFramesCounter_off(int checkpoint)
{
    std::ofstream file(m_frame_times_file, std::ios::out | std::ios::app);
    if (!file.is_open()) 
    {
        std::cout << "No .off file for frame times." << std::endl;
        return;
    }
    
    if (!m_dts.empty()) 
    {
        for (int i = 0; i < m_dts.size(); i++)
        {
            file 
            << checkpoint << "," 
            << i << "," 
            << m_dts[i]
            << "\n";
        }
    }
    
    file.close();
}

void Profiler::writeToProcessTime_off(std::string process_name)
{
    std::ofstream file(m_process_times_file, std::ios::out | std::ios::app);
    if (!file.is_open()) 
    {
        std::cout << "No .off file for processes times." << std::endl;
        return;
    }
    
    file << process_name << "," << m_process_total_time << "\n";
    file.close();
}