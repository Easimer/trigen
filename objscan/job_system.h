// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: simple, generic job system for paralellizing calculations
//

#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>

// A job source structure
// This is where threads fetch jobs from and publish their results to
// Job_Type is the struct/class that holds the details about one unit of work
// Result_Type is the data that a thread produces as a result.
template<typename Job_Type, typename Result_Type>
struct Job_Source {
    std::vector<std::thread> workers;
    
    std::mutex jobs_lock;
    std::vector<Job_Type> jobs;
    
    std::mutex results_lock;
    std::vector<Result_Type> results;
    
    using Job = Job_Type;
};

// Try to get a job unit from the queue
// @param jobs pointer to a Job_Source
// Returns true if the queue was not empty and fills
// in `job`.
template<typename JS, typename J>
static inline bool try_get_job(J& job, JS* jobs) {
    std::lock_guard G(jobs->jobs_lock);
    if(jobs->jobs.empty()) {
        return false;
    }
    
    job = jobs->jobs.back();
    jobs->jobs.pop_back();
    
    return true;
}

template<typename R, typename JS>
static inline void publish_result(R& res, JS* jobs) {
    std::lock_guard G(jobs->results_lock);
    jobs->results.push_back(res);
}