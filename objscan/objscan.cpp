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
#include <objscan.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

// TODO(danielm):
// - Assign particles to mesh vertices for animation
// - Find a data structure that speeds up neighbor search/connform
// - Multiresolution space sampling

#if 0
#define threading_dbgprint(fmt, i) printf(fmt, i)
#else
#define threading_dbgprint(fmt, i)
#endif

// Ray-triangle intersection
// Moeller-Trumbore algorithm
// Returns true on intersection and fills in `xp` and `t`, such that
// `origin + t * dir = xp`.
// Returns false otherwise.
static bool ray_triangle_intersect(
                                   glm::vec3& xp, float t,
                                   glm::vec3 const& origin, glm::vec3 const& dir,
                                   glm::vec3 const& v0, glm::vec3 const& v1, glm::vec3 const& v2
                                   ) {
    auto edge1 = v2 - v0;
    auto edge0 = v1 - v0;
    auto h = cross(dir, edge1);
    auto a = dot(edge0, h);
    if (-glm::epsilon<float>() < a && a < glm::epsilon<float>()) {
        return false;
    }
    
    auto f = 1.0f / a;
    auto s = origin - v0;
    auto u = f * dot(s, h);
    
    if(u < 0 || 1 < u) {
        return false;
    }
    
    auto q = cross(s, edge0);
    auto v = f * dot(dir, q);
    
    if(v < 0 || u + v > 1) {
        return false;
    }
    
    t = f * dot(edge1, q);
    if(t <= glm::epsilon<float>()) {
        return false;
    }
    
    xp = origin + t * dir;
    return true;
}

static glm::vec3 const& fetch_vertex_from_attrib_vertices(std::vector<tinyobj::real_t> const& vertices, int index) {
    // NOTE(danielm): this cast may not be the best idea
    return ((glm::vec3 const*)vertices.data())[index];
}

// Counts how many triangles does the given ray intersect.
static int count_triangles_intersected(
                                       glm::vec3 origin, glm::vec3 dir,
                                       tinyobj::attrib_t const& attrib,
                                       tinyobj::shape_t const& shape
                                       ) {
    int count = 0;
    auto triangle_count = shape.mesh.indices.size() / 3;
    auto& indices = shape.mesh.indices;
    
    for(int i = 0; i < triangle_count; i++) {
        auto i0 = indices[i * 3 + 0].vertex_index;
        auto i1 = indices[i * 3 + 1].vertex_index;
        auto i2 = indices[i * 3 + 2].vertex_index;
        auto& v0 = fetch_vertex_from_attrib_vertices(attrib.vertices, i0);
        auto& v1 = fetch_vertex_from_attrib_vertices(attrib.vertices, i1);
        auto& v2 = fetch_vertex_from_attrib_vertices(attrib.vertices, i2);
        
        glm::vec3 xp;
        float t = 0;
        if(ray_triangle_intersect(xp, t, origin, dir, v0, v1, v2)) {
            count++;
        }
    }
    
    return count;
}

// Determines whether the segment `origin -> origin + dir` intersects any triangles or not.
static bool intersects_any(
                           glm::vec3 origin, glm::vec3 dir,
                           tinyobj::attrib_t const& attrib,
                           tinyobj::shape_t const& shape
                           ) {
    auto triangle_count = shape.mesh.indices.size() / 3;
    auto& indices = shape.mesh.indices;
    
    for(int i = 0; i < triangle_count; i++) {
        auto i0 = indices[i * 3 + 0].vertex_index;
        auto i1 = indices[i * 3 + 1].vertex_index;
        auto i2 = indices[i * 3 + 2].vertex_index;
        auto& v0 = fetch_vertex_from_attrib_vertices(attrib.vertices, i0);
        auto& v1 = fetch_vertex_from_attrib_vertices(attrib.vertices, i1);
        auto& v2 = fetch_vertex_from_attrib_vertices(attrib.vertices, i2);
        
        glm::vec3 xp;
        float t = 0;
        if(ray_triangle_intersect(xp, t, origin, dir, v0, v1, v2)) {
            if(0 < t && t < 1) {
                return true;
            }
        }
    }
    
    return false;
}

static std::vector<glm::vec4> sample_points(
                                            float x_min, float x_max, float x_step,
                                            float y_min, float y_max, float y_step,
                                            float z_min, float z_max, float z_step,
                                            std::vector<tinyobj::shape_t> const& shapes,
                                            tinyobj::attrib_t const& attrib
                                            ) {
    std::vector<glm::vec4> points;
    auto dir = glm::vec3(1, 0, 0); // a random direction
    
    // Shoot a random ray from the point.
    // Count how many triangles it intersects.
    // If it's an odd number, it's on the inside.
    
    for (float sx = x_min; sx < x_max; sx += x_step) {
        for (float sy = y_min; sy < y_max; sy += y_step) {
            for (float sz = z_min; sz < z_max; sz += z_step) {
                auto origin = glm::vec3(sx, sy, sz);
                long long x_count = 0;
                
                for(auto const& shape : shapes) {
                    auto count = count_triangles_intersected(origin, dir, attrib, shape);
                    x_count += count;
                }
                
                if(x_count % 2 == 1) {
                    points.push_back({sx, sy, sz, 0});
                }
            }
        }
    }
    
    return points;
}

static std::vector<std::pair<int, int>> form_connections(
                                                         int offset, int count,
                                                         objscan_position const* positions, int N,
                                                         float step_x, float step_y, float step_z,
                                                         std::vector<tinyobj::shape_t> const& shapes,
                                                         tinyobj::attrib_t const& attrib
                                                         ) {
    std::vector<std::pair<int, int>> ret;
    
    for(int i = offset; i < offset + count; i++) {
        auto p = positions[i];
        for(int other = 0; other < N; other++) {
            if(i == other) continue;
            
            auto op = positions[other];
            
            auto dx = glm::abs(op.x - p.x);
            if(dx > 1.25 * step_x) {
                continue;
            }
            
            auto dy = glm::abs(op.y - p.y);
            if(dy > 1.25 * step_y) {
                continue;
            }
            
            auto dz = glm::abs(op.z - p.z);
            if(dz > 1.25 * step_z) {
                continue;
            }
            
            auto pv = glm::vec3(p.x, p.y, p.z);
            auto opv = glm::vec3(op.x, op.y, op.z);
            
            auto origin = pv;
            auto dir = opv - pv;
            
            bool no_intersection = true;
            
            for(auto& shape : shapes) {
                if(intersects_any(origin, dir, attrib, shape)) {
                    no_intersection = false;
                    break;
                }
            }
            
            if(no_intersection) {
                ret.push_back({i, other});
            }
        }
    }
    
    return ret;
}

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
static bool try_get_job(J& job, JS* jobs) {
    std::lock_guard G(jobs->jobs_lock);
    if(jobs->jobs.empty()) {
        return false;
    }
    
    job = jobs->jobs.front();
    jobs->jobs.pop();
    
    return true;
}

static void threadproc_sample_points(int threadIdx, 
                                     std::vector<tinyobj::shape_t> const& shapes, 
                                     tinyobj::attrib_t const& attrib,
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
                                 job.z_min, job.z_max, job.z_step, shapes, attrib);
        
        threading_dbgprint("objscan: thread=%d kind=sp finish\n", threadIdx);
        std::lock_guard G(jobs->results_lock);
        jobs->results.push_back(std::move(res));
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
                                std::vector<tinyobj::shape_t> const& shapes,
                                tinyobj::attrib_t const& attrib,
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
                                    shapes, attrib
                                    );
        threading_dbgprint("objscan: thread=%d kind=connform finish\n", threadIdx);
        std::lock_guard G(jobs->results_lock);
        jobs->results.push_back(std::move(res));
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
    float x_min = INFINITY, x_max = -INFINITY;
    float y_min = INFINITY, y_max = -INFINITY;
    float z_min = INFINITY, z_max = -INFINITY;
    
    auto pos_count = attrib.vertices.size() / 3;
    for (long long i = 0; i < pos_count; i++) {
        auto x = attrib.vertices[i * 3 + 0];
        auto y = attrib.vertices[i * 3 + 1];
        auto z = attrib.vertices[i * 3 + 2];
        
        if (x < x_min) x_min = x;
        if (x > x_max) x_max = x;
        if (y < y_min) y_min = y;
        if (y > y_max) y_max = y;
        if (z < z_min) z_min = z;
        if (z > z_max) z_max = z;
    }
    
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
    
    for(unsigned i = 0; i < thread_count; i++) {
        jobs.workers.emplace_back(std::thread(threadproc_sample_points, i, shapes, attrib, &jobs));
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
    
    for(unsigned i = 0; i < thread_count; i++) {
        connform.workers.emplace_back(std::thread(threadproc_connform, i, res->positions, total_particle_count, shapes, attrib, &connform));
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