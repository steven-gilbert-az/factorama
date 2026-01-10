#pragma once
#include <chrono>

namespace factorama
{
    class Stopwatch
    {
    public:
        Stopwatch() : is_running_(false), accumulated_time_(0.0) {}

        void start()
        {
            if (is_running_)
            {
                return;
            }
            is_running_ = true;
            start_time_ = std::chrono::high_resolution_clock::now();
        }

        void stop()
        {
            if (!is_running_)
            {
                return;
            }
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end_time - start_time_;
            accumulated_time_ += duration.count();
            is_running_ = false;
        }

        void reset()
        {
            is_running_ = false;
            accumulated_time_ = 0.0;
        }

        double elapsed() const
        {
            double total = accumulated_time_;
            if (is_running_)
            {
                auto current_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> duration = current_time - start_time_;
                total += duration.count();
            }
            return total;
        }

    private:
        bool is_running_;
        double accumulated_time_;
        std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
    };
}
