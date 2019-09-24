/**
 * ThreadPool
 * 
 * Used by: data loader
 * ....
 */

#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "../Logging.hpp"

namespace DLFS
{

class Task
{
public:
    Task();
    ~Task();

    virtual void Run() = 0;

private:
};

class ThreadPool
{
public:
    ThreadPool(unsigned int size)
    {
        for (unsigned int i = 0; i < size; i++)
        {
            m_threads.emplace_back(std::thread(
                [this]() {
                    this->Run();
                }));
        }
    }

    ~ThreadPool()
    {
        Shutdown();
    }

    void AddTask(Task *t)
    {
        m_tasks.push(t);
    }

    void Shutdown()
    {
        m_shutdown = true;
        m_cv.notify_all();
        for (auto &t : m_threads)
        {
            t.join();
        }

        LOG.INFO() << "Thread pool shutdown - all threads joined";
    }

    void Run()
    {
        while (!m_shutdown)
        {
            Task *task;
            {
                std::unique_lock<std::mutex> lock(m_queueMutex);
                m_cv.wait(lock, [this]() {
                    return (!m_tasks.empty() || m_shutdown);
                });

                if (m_shutdown)
                {
                    return;
                }

                // Acquired the lock and tasks are not empty
                task = m_tasks.front();
                m_tasks.pop();
            }
            m_cv.notify_one();

            // Run the task.
            task->Run();
        }
    }

private:
    bool m_shutdown{false};
    std::vector<std::thread> m_threads;
    std::queue<Task *> m_tasks;
    std::mutex m_queueMutex;
    std::condition_variable m_cv;
};

class ExampleTask : public Task
{
public:
    void Run()
    {
        LOG.INFO() << "Example task running.";
    }
};

} // namespace DLFS