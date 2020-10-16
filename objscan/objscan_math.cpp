// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#include "stdafx.h"
#include "objscan_math.h"
#include <numeric>
#include "mesh.h"
#include "intersection.h"

bool intersects_any(
                    glm::vec3 origin, glm::vec3 dir,
                    Mesh* mesh
                    ) {
    glm::vec3 xp;
    float t;
    auto triangle_count = mesh->triangles.size();
    auto dir_inv = glm::vec3(1 / dir.x, 1 / dir.y, 1 / dir.z);

    std::vector<int> filtered_triangle_indices;

    for (int i = 0; i < triangle_count; i++) {
        auto& bb = mesh->bounding_boxes[i];
        if (intersect::intersect_ray_aabb(origin, dir_inv, bb.min, bb.max)) {
            auto& tri = mesh->triangles[i];
            if (intersect::intersect_ray_triangle(xp, t, origin, dir, tri.v0, tri.v1, tri.v2)) {
                if (0 <= t && t <= 1) {
                    return true;
                }
            }
        }
    }

    return false;
}

std::vector<glm::vec4> sample_points(
                                     float x_min, float x_max, float x_step,
                                     float y_min, float y_max, float y_step,
                                     float z_min, float z_max, float z_step,
                                     Mesh* mesh
                                     ) {
    std::vector<glm::vec4> points;
    auto dir = glm::vec3(1, 0, 0); // a random direction
    auto dir_inv = glm::vec3(1, INFINITY, INFINITY);
    
    // Shoot a random ray from the point.
    // Count how many triangles it intersects.
    // If it's an odd number, it's on the inside.
    
    auto triangle_count = mesh->triangles.size();

    for (float sx = x_min; sx < x_max; sx += x_step) {
        for (float sy = y_min; sy < y_max; sy += y_step) {
            for (float sz = z_min; sz < z_max; sz += z_step) {
                auto origin = glm::vec3(sx, sy, sz);
                long long x_count = 0;

                glm::vec3 xp;
                float t;

                std::vector<int> filtered_triangle_indices;

                for (int i = 0; i < triangle_count; i++) {
                    auto& bb = mesh->bounding_boxes[i];
                    if (intersect::intersect_ray_aabb(origin, dir_inv, bb.min, bb.max)) {
                        filtered_triangle_indices.push_back(i);
                    }
                }

                for (auto idx : filtered_triangle_indices) {
                    auto& tri = mesh->triangles[idx];
                    if (intersect::intersect_ray_triangle(xp, t, origin, dir, tri.v0, tri.v1, tri.v2)) {
                        x_count++;
                    }
                }

                if(x_count % 2 == 1) {
                    points.push_back({sx, sy, sz, 0});
                }
            }
        }
    }
    
    return points;
}

std::vector<std::pair<int, int>> form_connections(
                                                  int offset, int count,
                                                  objscan_position const* positions, int N,
                                                  float step_x, float step_y, float step_z,
                                                  Mesh* mesh
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
            
            if (!intersects_any(origin, dir, mesh)) {
                ret.push_back({ i, other });
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