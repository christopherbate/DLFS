#ifndef _TIMER_H_
#define _TIMER_H_

#include <chrono>
#include <cmath>

class Timer
{
public:
    std::chrono::time_point<std::chrono::steady_clock> last_time;

    double tick()
    {
        auto time_now = std::chrono::steady_clock::now();

        unsigned int elapsed = 
            std::chrono::duration_cast<std::chrono::nanoseconds>(time_now - last_time).count();

        last_time = time_now;
        return (double)elapsed*1e-6;
    }

    double tick_us()
    {
        auto time_now = std::chrono::steady_clock::now();

        unsigned int elapsed = 
            std::chrono::duration_cast<std::chrono::nanoseconds>(time_now - last_time).count();

        last_time = time_now;
        return (double)elapsed*1e-3;
    }
};

#endif