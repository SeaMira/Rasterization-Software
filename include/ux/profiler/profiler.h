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
     * @param sphere_count the amount of spheres on the scene.
     * @param cylinder_count the amount of cylinders on the scene.
     * @param topMipmapLevel the top mipmap level for the HiZ pyramid.
     * @param seconds_timer the time the saving of frame times takes.
     */
    Profiler(Window& window, std::string frame_times_file, 
        std::string process_times_file, int sphere_count, int cylinder_count, int topMipmapLevel, double seconds_timer = 30.0);

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
     * 
     * @param checkpoint the checkpoint to be saved.
     * @param spheres_on_frustum the amount of spheres on the frustum.
     * @param visible_spheres the amount of visible spheres.
     * @param cylinders_on_frustum the amount of cylinders on the frustum.
     * @param visible_cylinders the amount of visible cylinders.
     */
    void updateProfiler(int checkpoint, int spheres_on_frustum, int visible_spheres, int cylinders_on_frustum, int visible_cylinders);

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
     * Writes the frame times to the file.
     * 
     */
    void writeToFramesCounter_off();

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
    int m_spheres_on_scene; ///< the amount of spheres on the scene.
    int m_cylinders_on_scene; ///< the amount of cylinders on the scene.
    int m_topMipmapLevel; ///< the top mipmap level of the hz pyramid.

    uint64_t m_frame_start_time; ///< the start time of saving frames.
    uint64_t m_frame_current_time; ///< the current time of the frame time saving.
    double m_last_frame_sec_dt; ///< the last frame time in seconds.
    double m_seconds_timer; ///< the time spent saving the frame times.
    std::vector<double> m_dts; ///< the frame times to be saved.
    std::vector<int> m_checkpoints; ///< vector with the traversed checkpoints for each frame.
    std::vector<int> m_spheres_on_frustum; ///< vector with amount of spheres on frustum for each frame.
    std::vector<int> m_visible_spheres; ///< vector with amount of visible spheres each frame.
    std::vector<int> m_cylinders_on_frustum; ///< vector with amount of cylinders on frustum for each frame.
    std::vector<int> m_visible_cylinders; ///< vector with amount of visible cylinders each frame.

    uint64_t m_process_start_time; ///< the start time of the process time saving.
    uint64_t m_process_total_time; ///< the total time of the process time saving.
    
    std::string m_frame_times_file; ///< the file to save the frame times.
    std::string m_process_times_file; ///< the file to save the process times.

    bool checking_frames = false; ///< flag for checking if the frames are being saved.
};

#endif // _PROFILER_H_