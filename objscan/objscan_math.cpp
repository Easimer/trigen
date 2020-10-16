// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#include "stdafx.h"
#include "objscan_math.h"
#include <numeric>

bool ray_triangle_intersect(
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

int count_triangles_intersected(
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

bool intersects_any(
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

std::vector<glm::vec4> sample_points(
                                     std::unique_ptr<ICompute>& compute,
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
    
    std::vector<int> indices;
    std::vector<int> hit;
    std::vector<glm::vec3> xp;
    std::vector<float> t;
    
    for(auto& shape : shapes) {
        auto& s_indices = shape.mesh.indices;
        
        for(auto& idx_tuple : s_indices) {
            indices.push_back(idx_tuple.vertex_index);
        }
    }
    
    auto triangle_count = indices.size() / 3;
    hit.resize(triangle_count);
    xp.resize(triangle_count);
    t.resize(triangle_count);
    
    for (float sx = x_min; sx < x_max; sx += x_step) {
        for (float sy = y_min; sy < y_max; sy += y_step) {
            for (float sz = z_min; sz < z_max; sz += z_step) {
                auto origin = glm::vec3(sx, sy, sz);
                
                compute->ray_triangles_intersect(triangle_count,
                                                 hit.data(), 
                                                 xp.data(),
                                                 t.data(),
                                                 origin, dir,
                                                 indices.data(),
                                                 (glm::vec3*)attrib.vertices.data());
                
                
                long long x_count = 0;
                x_count = std::accumulate(hit.cbegin(), hit.cend(), x_count);
                
                if(x_count % 2 == 1) {
                    points.push_back({sx, sy, sz, 0});
                }
            }
        }
    }
    
    return points;
}

std::vector<std::pair<int, int>> form_connections(
                                                  std::unique_ptr<ICompute>& compute,
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

void calculate_bounding_box(
                            float& x_min, float& y_min, float& z_min,
                            float& x_max, float& y_max, float& z_max,
                            tinyobj::attrib_t const& attrib) {
    x_min = INFINITY; x_max = -INFINITY;
    y_min = INFINITY; y_max = -INFINITY;
    z_min = INFINITY; z_max = -INFINITY;
    
    auto const pos_count = attrib.vertices.size() / 3;
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
}

class Compute_CPU : public ICompute {
    public:
    ~Compute_CPU() override = default;
    
    void ray_triangles_intersect(
                                 int N,
                                 int* out_hit,
                                 glm::vec3* out_xp,
                                 float* out_t,
                                 glm::vec3 const& origin, glm::vec3 const& dir,
                                 int const* vertex_indices,
                                 glm::vec3 const* vertex_positions
                                 ) override {
        for(int id = 0; id < N; id++) {
            auto i0 = vertex_indices[id * 3 + 0];
            auto i1 = vertex_indices[id * 3 + 1];
            auto i2 = vertex_indices[id * 3 + 2];
            auto& v0 = vertex_positions[i0];
            auto& v1 = vertex_positions[i1];
            auto& v2 = vertex_positions[i2];
            out_hit[id] = ray_triangle_intersect(out_xp[id], out_t[id], origin, dir, v0, v1, v2);
        }
    }
};

std::unique_ptr<ICompute> make_compute_backend(bool force_cpu) {
    return std::make_unique<Compute_CPU>();
}