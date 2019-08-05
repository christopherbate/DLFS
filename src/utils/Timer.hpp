#ifndef _TIMER_H_
#define _TIMER_H_

#include <chrono>
#include <cmath>

struct Timer
{
    std::chrono::time_point<std::chrono::steady_clock> last_time;

    float tick()
    {
        auto time_now = std::chrono::steady_clock::now();

        unsigned int elapsed = std::chrono::duration_cast<std::chrono::microseconds>(time_now - last_time).count();

        last_time = time_now;
        return elapsed/1000.0f;
    }
};

#endif