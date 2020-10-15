// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: converts triangle-based 3D models to a simulateable particle system
//

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

// Ray-triangle intersection
// Moeller-Trumbore algorithm
// Returns true on intersection and fills in `xp` and `t`, such that
// `origin + t * dir = xp`.
// Returns false otherwise.
static bool ray_triangle_intersect(
                                   glm::vec3& xp, float t,
                                   glm::vec3 origin, glm::vec3 dir,
                                   glm::vec3 v0, glm::vec3 v1, glm::vec3 v2
                                   ) {
    auto edge0 = v1 - v0;
    auto edge1 = v2 - v0;
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

static glm::vec3 fetch_vertex_from_attrib_vertices(std::vector<tinyobj::real_t> const& vertices, int index) {
    auto base = 3 * index;
    auto x = vertices[base + 0];
    auto y = vertices[base + 1];
    auto z = vertices[base + 2];
    
    return { x, y, z };
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
        auto v0 = fetch_vertex_from_attrib_vertices(attrib.vertices, i0);
        auto v1 = fetch_vertex_from_attrib_vertices(attrib.vertices, i1);
        auto v2 = fetch_vertex_from_attrib_vertices(attrib.vertices, i2);
        
        glm::vec3 xp;
        float t = 0;
        if(ray_triangle_intersect(xp, t, origin, dir, v0, v1, v2)) {
            count++;
        }
    }
    
    return count;
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
                    //printf("Point %f %f %f is inside!\n", sx, sy, sz);
                    points.push_back({sx, sy, sz, 0});
                }
            }
        }
    }
    
    return points;
}

struct job_t {
    float x_min, x_max, x_step;
    float y_min, y_max, y_step;
    float z_min, z_max, z_step;
};

struct job_src_t {
    std::vector<std::thread> workers;
    
    std::mutex jobs_lock;
    std::queue<job_t> jobs;
    
    std::mutex results_lock;
    std::vector<std::vector<glm::vec4>> results;
};

static void threadproc_sample_points(int threadIdx, 
                                     std::vector<tinyobj::shape_t> const& shapes, 
                                     tinyobj::attrib_t const& attrib,
                                     job_src_t* jobs) {
    for(;;) {
        job_t job;
        {
            std::lock_guard G(jobs->jobs_lock);
            if(jobs->jobs.empty()) {
                printf("objscan: thread=%d exit\n", threadIdx);
                return;
            }
            job = jobs->jobs.front();
            jobs->jobs.pop();
        }
        
        printf("objscan: thread=%d ready\n", threadIdx);
        auto res = sample_points(
                                 job.x_min, job.x_max, job.x_step,
                                 job.y_min, job.y_max, job.y_step,
                                 job.z_min, job.z_max, job.z_step, shapes, attrib);
        
        printf("objscan: thread=%d finish\n", threadIdx);
        std::lock_guard G(jobs->results_lock);
        jobs->results.push_back(std::move(res));
        printf("objscan: thread=%d publish\n", threadIdx);
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
    
    printf("Model bounding box: [%f %f %f], [%f %f %f]\n", x_min, y_min, z_min, x_max, y_max, z_max);
    
    auto subdivisions = 64.0f;
    auto x_step = (x_max - x_min) / subdivisions;
    auto y_step = (y_max - y_min) / subdivisions;
    auto z_step = (z_max - z_min) / subdivisions;
    
    printf("Subdivisions: %f\nSteps: %f %f %f\n", subdivisions, x_step, y_step, z_step);
    
    auto thread_count = std::thread::hardware_concurrency();
    auto thread_count_cubed = thread_count * thread_count * thread_count;
    auto blockDim_x = (x_max - x_min) / thread_count;
    auto blockDim_y = (y_max - y_min) / thread_count;
    auto blockDim_z = (z_max - z_min) / thread_count;
    job_src_t jobs;
    
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
    
    res->connection_count = 0;
    
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