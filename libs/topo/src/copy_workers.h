// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <vector>

#include <Tracy.hpp>

namespace topo {
class Copy_Workers {
public:
    struct Task {
        void const *source;
        void *destination;
        size_t size;
    };

    Copy_Workers()
        : Copy_Workers(std::thread::hardware_concurrency()) { }

    Copy_Workers(unsigned numWorkers) : _threads(numWorkers) {
        for (unsigned worker = 0; worker < numWorkers; worker++) {
            _threads[worker] = std::thread([&, this, worker]() {
                char threadName[64];
                snprintf(threadName, 63, "memcpy worker %p::%u", this, worker);
                threadName[63] = '\0';
                tracy::SetThreadName(threadName);

                std::optional<Task> task;
                do {
                    task = this->Pop();
                    if (task) {
                        ZoneScoped;
                        memcpy(task->destination, task->source, task->size);

                        std::unique_lock G(_signalsLock);
                        _completionSignals.push({});
                        _signalsCV.notify_one();
                    }
                } while (task);
            });
        }
    }

    ~Copy_Workers() {
        _shutdown = true;
        _tasksCV.notify_all();
        for (auto& thread : _threads) {
            thread.join();
        }
    }

    void
    Push(unsigned numTasks, Task const *tasks) {
        std::unique_lock G(_tasksLock);

        for (unsigned i = 0; i < numTasks; i++) {
            _tasks.push(tasks[i]);
            _tasksCV.notify_one();
        }
    }

    std::optional<Task>
    Pop() {
        std::unique_lock G(_tasksLock);
        while (_tasks.empty() && !_shutdown) {
            _tasksCV.wait(G);
        }

        if (_shutdown) {
            return std::nullopt;
        }

        std::optional<Task> ret = _tasks.front();
        _tasks.pop();

        return ret;
    }

    void
    WaitTasks(int numTasks) {
        while (numTasks > 0) {
            std::unique_lock G(_signalsLock);
            while (_completionSignals.empty()) {
                _signalsCV.wait(G);
            }
            _completionSignals.pop();
            numTasks--;
        }
    }

private:
    std::vector<std::thread> _threads;

    std::atomic<bool> _shutdown;
    std::queue<Task> _tasks;
    std::mutex _tasksLock;
    std::condition_variable _tasksCV;

    struct Empty { };

    std::queue<Empty> _completionSignals;
    std::mutex _signalsLock;
    std::condition_variable _signalsCV;
};
}
