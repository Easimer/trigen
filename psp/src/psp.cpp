// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#include <cassert>
#include <cfloat>
#include <deque>
#include <unordered_set>
#include <algorithm>

#include <glm/geometric.hpp>
#include <glm/gtx/component_wise.hpp>

#include <psp/psp.h>
#include <numeric>
#include <optional>

using Triangle_ID = size_t;

static glm::vec3 const dirvec[6] = { {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1} };
static glm::vec3 const projector_vec[6] = { {0, 1, 1}, {0, 1, 1}, {1, 0, 1}, {1, 0, 1}, {1, 1, 0}, {1, 1, 0} };
static glm::ivec2 const selector_vec[6] = { {1, 2}, {1, 2}, {0, 2}, {0, 2}, {0, 1}, {0, 1} };

struct Chart {
    int direction;
    std::vector<Triangle_ID> triangles;
};

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

static std::vector<Chart> chart(PSP::Mesh &mesh) {
    auto charts = find_charts(mesh);
    std::sort(charts.begin(), charts.end(), [&](Chart const &lhs, Chart const &rhs) -> bool {
        return lhs.triangles.size() < rhs.triangles.size();
    });

    assign_debug_vertex_colors(mesh, charts);

    return charts;
}

static void project_charts(PSP::Mesh &mesh, std::vector<Chart> &charts) {
    std::vector<std::optional<glm::vec2>> uv_tmp;
    uv_tmp.resize(mesh.position.size());

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
            // per-component multiplication
            for (int v = 0; v < 3; v++) {
                auto vtxIdx = mesh.elements[baseVtxIdx + v];

                if (uv_tmp[vtxIdx].has_value()) {
                    continue;
                }

                auto p = mesh.position[vtxIdx] * proj;

                for (int c = 0; c < 3; c++) {
                }

                glm::vec2 uv;
                for (int idx2 = 0; idx2 < 2; idx2++) {
                    auto idx3 = selector_vec[chart.direction][idx2];
                    min[idx2] = glm::min(min[idx2], p[idx3]);
                    max[idx2] = glm::max(max[idx2], p[idx3]);
                    uv[idx2] = p[idx3];
                }
                uv_tmp[vtxIdx] = uv;
            }
        }

        // After we have all the UV coords in model space, we map them to normalized chart space.
        // UV' = (UV + min) / (max - min)
        for (auto tri : chart.triangles) {
            auto baseVtxIdx = tri * 3;
            // per-component multiplication
            for (int v = 0; v < 3; v++) {
                auto vtxIdx = mesh.elements[baseVtxIdx + v];
                assert(uv_tmp[vtxIdx].has_value());

                auto &uv = uv_tmp[vtxIdx].value();

                uv = (uv + min) / (max - min);

                assert(0.0f <= uv[0] && uv[0] <= 1.0f);
                assert(0.0f <= uv[1] && uv[1] <= 1.0f);
            }
        }
    }

    assert(uv_tmp.size() == mesh.position.size());
    mesh.uv.resize(uv_tmp.size());
    for (size_t i = 0; i < uv_tmp.size(); i++) {
        assert(uv_tmp[i].has_value());
        mesh.uv[i] = uv_tmp[i].value();
    }
}

int PSP::paint(/* out */ Material &material, /* inout */ Mesh &mesh) {
    // Assign UV coordinates
    auto charts = chart(mesh);
    project_charts(mesh, charts);
    // Setup particle system
    // Simulate particles:
    //  - calculate next position based on velocity
    //  - check collisions; for all triangle: collision(triangle, ray(next, current))
    //    - on collision, apply brush to the material
    return 0;
}