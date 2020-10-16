// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: converts triangle-based 3D models to a simulateable particle system
//

#include "stdafx.h"
#include <cstdio>
#include <fstream>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include <objscan.h>
#include "objscan_math.h"
#include "mesh.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

// TODO(danielm):
// - Assign particles to mesh vertices for animation
// - Find a data structure that speeds up neighbor search/connform
// - Multiresolution space sampling

#if 1
#define threading_dbgprint(fmt, i) printf(fmt, i)
#else
#define threading_dbgprint(fmt, i)
#endif


template<typename Job_Type, typename Result_Type>
struct Job_Source {
    std::vector<std::thread> workers;
    
    std::mutex jobs_lock;
    std::queue<Job_Type> jobs;
    
    std::mutex results_lock;
    std::vector<Result_Type> results;
};

struct Space_Sampling_Job {
    float x_min, x_max, x_step;
    float y_min, y_max, y_step;
    float z_min, z_max, z_step;
};
using Space_Sampling_Jobs = Job_Source<Space_Sampling_Job, std::vector<glm::vec4>>;

template<typename JS, typename J>
static inline bool try_get_job(J& job, JS* jobs) {
    std::lock_guard G(jobs->jobs_lock);
    if(jobs->jobs.empty()) {
        return false;
    }
    
    job = jobs->jobs.front();
    jobs->jobs.pop();
    
    return true;
}

template<typename R, typename JS>
static inline void publish_result(R& res, JS* jobs) {
    std::lock_guard G(jobs->results_lock);
    jobs->results.push_back(std::move(res));
}

static void threadproc_sample_points(int threadIdx, 
                                     Mesh* mesh,
                                     Space_Sampling_Jobs* jobs) {
    for(;;) {
        Space_Sampling_Job job;
        if(!try_get_job(job, jobs)) {
            threading_dbgprint("objscan: thread=%d kind=sp exit\n", threadIdx);
            return;
        }
        
        threading_dbgprint("objscan: thread=%d kind=sp ready\n", threadIdx);
        auto res = sample_points(
                                 job.x_min, job.x_max, job.x_step,
                                 job.y_min, job.y_max, job.y_step,
                                 job.z_min, job.z_max, job.z_step, mesh);
        
        threading_dbgprint("objscan: thread=%d kind=sp finish\n", threadIdx);
        
        publish_result(res, jobs);
        threading_dbgprint("objscan: thread=%d kind=sp publish\n", threadIdx);
    }
}

struct Connection_Forming_Job {
    int offset, count;
    float x_step, y_step, z_step;
};
using Connection_Forming_Jobs = Job_Source<Connection_Forming_Job, std::vector<std::pair<int, int>>>;

static void threadproc_connform(int threadIdx,
                                objscan_position const* positions, int N,
                                Mesh* mesh,
                                Connection_Forming_Jobs* jobs) {
    for(;;) {
        Connection_Forming_Job job;
        
        if(!try_get_job(job, jobs)) {
            threading_dbgprint("objscan: thread=%d kind=connform exit\n", threadIdx);
            return;
        }
        
        threading_dbgprint("objscan: thread=%d kind=connform ready\n", threadIdx);
        auto res = form_connections(
                                    job.offset, job.count,
                                    positions, N,
                                    job.x_step, job.y_step, job.z_step,
                                    mesh
                                    );
        threading_dbgprint("objscan: thread=%d kind=connform finish\n", threadIdx);
        publish_result(res, jobs);
        threading_dbgprint("objscan: thread=%d kind=connform publish\n", threadIdx);
    }
}

bool objscan_from_obj_file(objscan_result* res, char const* path) {
    if (res == NULL || path == NULL) {
        return false;
    }
    
    res->positions = NULL;
    res->connections = NULL;
    
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;
    
    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &err, path)) {
        return false;
    }
    
    // Calculate bounding box
    float x_min, y_min, z_min;
    float x_max, y_max, z_max;
    calculate_bounding_box(x_min, y_min, z_min, x_max, y_max, z_max, attrib);

    auto mesh = create_mesh(shapes, attrib);
    
    float subdivisions = 32.0f;
    if(res->extra != NULL) {
        subdivisions = res->extra->subdivisions;
    }
    
    auto const x_step = (x_max - x_min) / subdivisions;
    auto const y_step = (y_max - y_min) / subdivisions;
    auto const z_step = (z_max - z_min) / subdivisions;
    
    auto thread_count = std::thread::hardware_concurrency();
    
    if(res->extra != NULL) {
        res->extra->step_x = x_step;
        res->extra->step_y = y_step;
        res->extra->step_z = z_step;
        res->extra->bb_min = { x_min, y_min, z_min };
        res->extra->bb_max = { x_max, y_max, z_max };
        res->extra->threads_used = thread_count;
    }
    
    auto blockDim_x = (x_max - x_min) / thread_count;
    auto blockDim_y = (y_max - y_min) / thread_count;
    auto blockDim_z = (z_max - z_min) / thread_count;
    Space_Sampling_Jobs jobs;
    
    for(unsigned x = 0; x < thread_count; x++) {
        for(unsigned y = 0; y < thread_count; y++) {
            for(unsigned z = 0; z < thread_count; z++) {
                jobs.jobs.push({});
                auto& job = jobs.jobs.back();
                
                job.x_min = x_min + x * blockDim_x;
                job.x_max = job.x_min + blockDim_x;
                job.x_step = x_step;
                
                job.y_min = y_min + y * blockDim_y;
                job.y_max = job.y_min + blockDim_y;
                job.y_step = y_step;
                
                job.z_min = z_min + z * blockDim_z;
                job.z_max = job.z_min + blockDim_z;
                job.z_step = z_step;
            }
        }
    }
    
    for(int i = 0; i < thread_count; i++) {
        jobs.workers.emplace_back(std::thread(threadproc_sample_points, i, &mesh, &jobs));
    }
    
    for(auto& worker : jobs.workers) {
        worker.join();
    }
    
    long long total_particle_count = 0;
    for(auto& result : jobs.results) {
        total_particle_count += result.size();
    }
    
    auto out_positions = new objscan_position[total_particle_count];
    
    int idx = 0;
    
    for(auto& result : jobs.results) {
        for(auto& v : result) {
            out_positions[idx++] = { v.x, v.y, v.z, 0 };
        }
    }
    
    res->particle_count = total_particle_count;
    res->positions = out_positions;

    if (total_particle_count == 0) {
        return false;
    }
    
    // Form connections between particles
    Connection_Forming_Jobs connform;
    
    auto connform_batch_size = total_particle_count / thread_count;
    auto connform_remains = total_particle_count;
    auto offset = 0;
    
    while(connform_remains >= connform_batch_size) {
        connform.jobs.push({});
        auto& job = connform.jobs.back();
        
        job.offset = offset;
        job.count = connform_batch_size;
        job.x_step = x_step;
        job.y_step = y_step;
        job.z_step = z_step;
        
        offset += connform_batch_size;
        connform_remains -= connform_batch_size;
    }
    
    if(connform_remains > 0) {
        connform.jobs.push({});
        auto& job = connform.jobs.back();
        
        job.offset = offset;
        job.count = connform_remains;
        job.x_step = x_step;
        job.y_step = y_step;
        job.z_step = z_step;
    }
    
    for(int i = 0; i < thread_count; i++) {
        connform.workers.emplace_back(std::thread(threadproc_connform, i, res->positions, total_particle_count, &mesh, &connform));
    }
    
    for(auto& worker : connform.workers) {
        worker.join();
    }
    
    long long total_connections = 0;
    for(auto& result : connform.results) {
        total_connections += result.size();
    }
    
    auto connections = new objscan_connection[total_connections];
    
    idx = 0;
    for(auto& result : connform.results) {
        for(auto p : result) {
            connections[idx++] = { p.first, p.second };
        }
    }
    
    res->connection_count = total_connections;
    res->connections = connections;
    
    return true;
}

void objscan_free_result(objscan_result* res) {
    if (res == NULL) {
        return;
    }
    
    if (res->connections != NULL) {
        delete[] res->connections;
    }
    res->connections = NULL;
    
    if (res->positions != NULL) {
        delete[] res->positions;
    }
    res->positions = NULL;
    
    res->connection_count = res->particle_count = 0;
}