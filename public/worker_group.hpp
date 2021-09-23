// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#pragma once

#include <functional>
#include <deque>
#include <thread>
#include <optional>
#include <boost/thread/thread.hpp>

class Worker_Group {
public:
    using Task = std::optional<std::function<void()>>;

    Worker_Group() {
        _num_workers = std::thread::hardware_concurrency();
        for (unsigned i = 0; i < _num_workers; i++) {
            _threads.create_thread(std::bind(threadfunc, this));
        }
    }

    ~Worker_Group() {
        for (unsigned i = 0; i < _num_workers; i++) {
            emplace_task(Task());
        }

        _threads.join_all();
    }

    void emplace_task(Task &&task) {
        _task_queue_lock.lock();
        _task_queue.push_back(std::move(task));
        _task_queue_lock.unlock();
        _sem_ready_tasks.release();
    }

    void wait() {
        _sem_ready_workers.wait_value(_num_workers);
    }

private:
    class Semaphore {
        volatile unsigned _count = 0;
        boost::mutex _mutex;
        boost::condition_variable _cv;

    public:
        void release() {
            boost::unique_lock<boost::mutex> lock(_mutex);
            _count++;
            _cv.notify_one();
        }

        void wait() {
            boost::unique_lock<boost::mutex> lock(_mutex);
            while (_count == 0) {
                _cv.wait(lock);
            }
            _count--;
        }

        void wait_value(unsigned value) {
            boost::unique_lock<boost::mutex> lock(_mutex);
            while (_count < value) {
                _cv.wait(lock);
            }
        }
    };

    static void threadfunc(Worker_Group *group) {
        bool stop = false;
        group->_sem_ready_workers.release();

        while (!stop) {
            group->_sem_ready_tasks.wait();
            group->_task_queue_lock.lock();
            Task task = std::move(group->_task_queue.front());
            group->_task_queue.pop_front();
            group->_task_queue_lock.unlock();
            group->_sem_ready_workers.wait();
            if (task.has_value()) {
                auto &func = *task;
                func();
            } else {
                stop = true;
            }
            group->_sem_ready_workers.release();
        }
    }

    unsigned _num_workers;
    boost::thread_group _threads;
    boost::mutex _task_queue_lock;
    std::deque<Task> _task_queue;

    Semaphore _sem_ready_tasks;
    Semaphore _sem_ready_workers;
};
