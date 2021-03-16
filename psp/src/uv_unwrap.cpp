// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#include "uv_unwrap.h"

#include <psp/psp.h>

#include <cstdio>
#include <cassert>
#include <cfloat>
#include <cstdio>
#include <deque>
#include <unordered_set>
#include <algorithm>
#include <unordered_map>
#include <deque>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/geometric.hpp>
#include <glm/gtx/component_wise.hpp>

#include <psp/psp.h>
#include <numeric>
#include <optional>

static glm::vec3 const dirvec[6] = { {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1} };
static glm::vec3 const projector_vec[6] = { {0, 1, 1}, {0, 1, 1}, {1, 0, 1}, {1, 0, 1}, {1, 1, 0}, {1, 1, 0} };
static glm::ivec2 const selector_vec[6] = { {1, 2}, {1, 2}, {0, 2}, {0, 2}, {0, 1}, {0, 1} };

static int get_triangle_direction(PSP::Mesh const &mesh, Triangle_ID tid) {

    glm::vec3 tri_normal(0, 0, 0);
    for (int j = 0; j < 3; j++) {
        auto v = mesh.elements[tid * 3 + j];
        tri_normal += mesh.normal[v];
    }
    tri_normal = glm::normalize(tri_normal);

    int min_i = 0;
    float min_diff = FLT_MAX;

    for (int i = 0; i < 6; i++) {
        auto d = glm::abs(1 - glm::dot(tri_normal, dirvec[i]));
        if (d < min_diff) {
            min_i = i;
            min_diff = d;
        }
    }

    return min_i;
}

static bool check_duplicate_vertices(PSP::Mesh const &mesh, std::vector<Chart> const &charts) {
    std::unordered_map<size_t, Chart*> vertex_indices;

    for (auto &chart : charts) {
        for (auto &triangle_id : chart.triangles) {
            auto base_vertex_index = triangle_id * 3;
            for (int v = 0; v < 3; v++) {
                auto element = mesh.elements[base_vertex_index + v];
                if (vertex_indices.count(element)) {
                    auto other_chart_ptr = vertex_indices[element];
                    fprintf(stderr, "Chart %p contains vertex %zu already contained by chart %p\n", &chart, element, other_chart_ptr);
                    return false;
                }
            }
        }
    }

    return true;
}

static std::vector<Chart> find_charts(PSP::Mesh const &mesh) {
    std::vector<Chart> charts;
    std::unordered_set<size_t> triangles;

    auto vertex_count = mesh.elements.size();
    auto triangle_count = vertex_count / 3;
    for (size_t i = 0; i < triangle_count; i++) {
        triangles.insert(i);
    }

    while (!triangles.empty()) {
        std::deque<Triangle_ID> tri_queue;
        Chart chart;
        // Pick a random triangle
        auto init_tri_it = triangles.begin();
        auto init_tri = *init_tri_it;
        triangles.erase(init_tri_it);
        // Determine direction
        auto const dir = get_triangle_direction(mesh, init_tri);
        chart.direction = dir;
        // Add it to the queue
        tri_queue.push_back(init_tri);

        while (!tri_queue.empty()) {
            // Pop off a triangle
            auto tid = tri_queue.front();
            tri_queue.pop_front();

            // Determinate direction
            auto tdir = get_triangle_direction(mesh, tid);
            if (tdir != dir) {
                // Isn't facing the same way as the current chart
                triangles.insert(tid);
                continue;
            }

            // Add it to the chart
            chart.triangles.push_back(tid);

            // Find neighboring triangles by looking at the vertex IDs
            size_t vid[3];
            for (int vid_off = 0; vid_off < 3; vid_off++) {
                vid[vid_off] = mesh.elements[tid * 3 + vid_off];
            }

            // A triangle neighbors the current one if they share at least a single vertex
            auto it = triangles.begin();
            while (it != triangles.end()) {
                bool is_neighbor = false;
                auto otid = *it;

                for (int vid_off_cur = 0; vid_off_cur < 3; vid_off_cur++) {
                    for (int vid_off = 0; vid_off < 3; vid_off++) {
                        if (vid[vid_off_cur] == mesh.elements[otid * 3 + vid_off]) {
                            is_neighbor = true;
                            break;
                        }
                    }

                    if (is_neighbor) {
                        break;
                    }
                }

                if (is_neighbor) {
                    tri_queue.push_back(*it);
                    triangles.erase(it++);
                } else {
                    ++it;
                }
            }
        }

        charts.push_back(std::move(chart));
    }

    assert(check_duplicate_vertices(mesh, charts));

    return charts;
}

static float hue_to_rgb(float p, float q, float t) {
    while (t < 0) t += 1;
    while (t > 1) t -= 1;

    if (t < 1 / 6.0f) {
        return p + (q - p) * 6 * t;
    }

    if (t < 1 / 2.0f) {
        return q;
    }

    if (t < 2 / 3.0f) {
        return p + (q - p) * (2 / 3.0f - t) * 6;
    }

    return p;
}

static void generate_colors(size_t N, std::vector<glm::u8vec3> &vec) {
    for (size_t i = 0; i < N; i++) {
        glm::vec3 col;
        float hue = i * (1 / (float)N);
        float saturation = 0.75f + 0.25f * ((rand() % 100) / 100.0f);
        float lightness = 0.5f + 0.5f * ((rand() % 100) / 100.0f);

        float q = lightness < 0.5f ? lightness * (1 + saturation) : lightness + saturation - lightness * saturation;
        float p = 2 * lightness - q;

        col.r = hue_to_rgb(p, q, hue + 1 / 3.0f);
        col.g = hue_to_rgb(p, q, hue);
        col.b = hue_to_rgb(p, q, hue - 1 / 3.0f);

        vec.push_back({ col.r * 255, col.g * 255, col.b * 255 });
    }
}

static void assign_debug_vertex_colors(PSP::Mesh &mesh, std::vector<Chart> const &charts) {
    std::vector<glm::u8vec3> colors;
    generate_colors(charts.size(), colors);

    mesh.chart_debug_color.resize(mesh.position.size());

    for (size_t i = 0; i < charts.size(); i++) {
        for (auto triangle : charts[i].triangles) {
            mesh.chart_debug_color[mesh.elements[triangle * 3 + 0]] = colors[i];
            mesh.chart_debug_color[mesh.elements[triangle * 3 + 1]] = colors[i];
            mesh.chart_debug_color[mesh.elements[triangle * 3 + 2]] = colors[i];
        }
    }
}

static float area_of_triangle(glm::vec3 const &A, glm::vec3 const &B, glm::vec3 const &C) {
    auto AB = B - A;
    auto AC = C - A;
    return length(cross(AB, AC)) / 2;
}

static float area_of_chart(PSP::Mesh const &mesh, Chart const &chart) {
    float total = 0;
    for (auto triangle_id : chart.triangles) {
        auto base_idx = triangle_id * 3;
        auto area = area_of_triangle(
            mesh.position[mesh.elements[base_idx + 0]],
            mesh.position[mesh.elements[base_idx + 1]],
            mesh.position[mesh.elements[base_idx + 2]]
        );
        total += area;
    }
    return total;
}

void sort_charts_by_area_descending(PSP::Mesh const &mesh, std::vector<Chart> &charts) {
    auto pred = [mesh](Chart const &lhs, Chart const &rhs) {
        return area_of_chart(mesh, lhs) > area_of_chart(mesh, rhs);
    };
    sort(charts.begin(), charts.end(), pred);
}

std::vector<Chart> chart(PSP::Mesh &mesh) {
    auto charts = find_charts(mesh);
    sort_charts_by_area_descending(mesh, charts);

    assign_debug_vertex_colors(mesh, charts);

    return charts;
}

void project_charts(PSP::Mesh &mesh, std::vector<Chart> &charts) {
    std::vector<std::optional<glm::vec2>> uv_tmp;
    std::vector<std::optional<glm::vec2>> uv_normalized;
    uv_tmp.resize(mesh.elements.size());
    uv_normalized.resize(mesh.elements.size());

    for (auto &chart : charts) {
        glm::vec2 min(FLT_MAX, FLT_MAX);
        glm::vec2 max(FLT_MIN, FLT_MIN);

        assert(chart.direction < sizeof(projector_vec) / sizeof(projector_vec[0]));

        // Project the vertices in the current chart to a plane.
        // This plane's normal is the vector pointing in the direction of the chart.
        //
        // Resulting vectors are still in model space.
        auto proj = projector_vec[chart.direction];
        for (auto tri : chart.triangles) {
            auto baseVtxIdx = tri * 3;
            for (int v = 0; v < 3; v++) {
                auto elementIdx = baseVtxIdx + v;
                auto vtxIdx = mesh.elements[elementIdx];

                assert(!uv_tmp[elementIdx].has_value());

                // NOTE(danielm): component-wise vector product
                auto p = mesh.position[vtxIdx] * proj;

                // Calculate temporary UV coordinate
                glm::vec2 uv;
                for (int idx2 = 0; idx2 < 2; idx2++) {
                    auto idx3 = selector_vec[chart.direction][idx2];
                    min[idx2] = glm::min(min[idx2], p[idx3]);
                    max[idx2] = glm::max(max[idx2], p[idx3]);
                    uv[idx2] = p[idx3];
                }
                uv_tmp[elementIdx] = uv;
            }
        }

        auto size = max - min;
        if (fabs(size.x) < FLT_EPSILON || fabs(size.y) < FLT_EPSILON) {
            // Degenerate chart with zero area, add a little bit of size to it
            max += glm::vec2 { 0.1f, 0.1f };
        }

        // After we have all the UV coords in model space, we map them to normalized chart space.
        // UV' = (UV - min) / (max - min)
        for (auto tri : chart.triangles) {
            auto baseVtxIdx = tri * 3;
            // per-component multiplication
            for (int v = 0; v < 3; v++) {
                auto elementIdx = baseVtxIdx + v;
                auto vtxIdx = mesh.elements[baseVtxIdx + v];

                // Prevent UV coords from being normalized multiple times
                assert(!uv_normalized[elementIdx].has_value());

                assert(uv_tmp[elementIdx].has_value());

                auto uv = uv_tmp[elementIdx].value();
                assert(min.x <= uv.x && uv.x <= max.x);
                assert(min.y <= uv.y && uv.y <= max.y);

                uv_normalized[elementIdx] = (uv - min) / (max - min);

                auto uv_n = uv_normalized[elementIdx].value();

                assert(0.0f <= uv_n[0] && uv_n[0] <= 1.0f);
                assert(0.0f <= uv_n[1] && uv_n[1] <= 1.0f);
            }
        }
    }

    assert(uv_normalized.size() == mesh.elements.size());
    mesh.uv.resize(uv_normalized.size());
    for (size_t i = 0; i < uv_normalized.size(); i++) {
        assert(uv_normalized[i].has_value());
        mesh.uv[i] = uv_normalized[i].value();
    }
}

struct Quad {
    glm::vec2 min, max;
};

/*
iterator divide_quad(m, M, i_limit):
  Q := queue()
  Q.enqueue(([0, 0], [1, 1]))
  i := 0
  while Q is not empty do
    quad := Q.dequeue()
    yield Q0(quad)
    yield Q1(quad)
    i := i + 2
    if i < i_limit:
      Q.enqueue(Q2(quad))
      Q.enqueue(Q3(quad))
*/
/*
 * Divides the UV space up among the charts in the manner of a breadth-first
 * search.
 * TODO(danielm): better explanation
 */
static std::vector<Quad> divide_quad(size_t size_limit) {
    std::vector<Quad> ret;
    std::deque<Quad> queue;

    queue.push_back({ {0, 0}, {1, 1} });

    while (!queue.empty()) {
        auto quad = queue.front();
        queue.pop_front();
        glm::vec2 half_width = { (quad.max - quad.min).x / 2, 0 };
        glm::vec2 half_height = { 0, (quad.max - quad.min).y / 2 };
        auto center = quad.min + half_width + half_height;

        // Q0: (top-left, center)
        ret.push_back({ quad.min, center });
        // Q1: (top-center, center-right)
        ret.push_back({ quad.min + half_width, center + half_width });

        if (ret.size() < size_limit) {
            // Q2: (center-left, bottom-center)
            queue.push_back({ quad.min + half_height, center + half_height });
            // Q3: (center, bottom-right)
            queue.push_back({ center, quad.max });
        } else {
            break;
        }
    }

    return ret;
}

static bool is_inside(glm::vec2 x, glm::vec2 m, glm::vec2 M) {
    return (m.x <= x.x && x.x <= M.x && m.y <= x.y && x.y <= M.y);
}

void divide_quad_among_charts(PSP::Mesh &mesh, std::vector<Chart> const &charts) {
    // Generate enough quads for every chart
    auto quads = divide_quad(charts.size());
    // `charts` is assumed to be sorted by total surface area (desc.)
    // so the biggest chart gets the biggest quad in the UV space
    std::vector<std::optional<glm::vec2>> new_uvs;
    new_uvs.resize(mesh.elements.size());

    for (int chart_idx = 0; chart_idx < charts.size(); chart_idx++) {
        auto &quad = quads[chart_idx];
        auto &chart = charts[chart_idx];
        // UV coordinates are currently in chart space, so we need to transform them
        // into UV space
        auto const quad_extent = quad.max - quad.min;
        for (auto &triangle_id : chart.triangles) {
            auto base_vertex_idx = triangle_id * 3;
            for (int v = 0; v < 3; v++) {
                auto elementIdx = base_vertex_idx + v;
                assert(!new_uvs[elementIdx].has_value());

                auto &uv = mesh.uv[elementIdx];
                // Scale UV coordinate according to the quad extent then move
                // it into quad space
                auto uv_in_quad = quad.min + uv * quad_extent;

                assert(is_inside(uv_in_quad, quad.min, quad.max));

                new_uvs[elementIdx] = uv_in_quad;
            }
        }
    }

    for (size_t i = 0; i < new_uvs.size(); i++) {
        mesh.uv[i] = new_uvs[i].value();
    }
}

