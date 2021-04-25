// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#include <cfloat>
#include <tuple>
#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <marching_cubes.h>
#include "mc33/MC33.h"

using namespace marching_cubes;

static void calc_bounds(glm::vec3 &min, glm::vec3 &max, std::vector<metaball> const &metaballs) {
    min = { FLT_MAX, FLT_MAX, FLT_MAX };
    max = { FLT_MIN, FLT_MIN, FLT_MIN };

    for (auto &mb : metaballs) {
        for (int i = 0; i < 3; i++) {
            min[i] = glm::min(min[i], mb.position[i] - mb.radius - 1.5f);
            max[i] = glm::max(max[i], mb.position[i] + mb.radius + 1.5f);
        }
    }
}

struct grid_iterator {
    unsigned x, y, z;
    unsigned lx, ly, lz;

    grid_iterator(unsigned subdiv) : grid_iterator(subdiv, subdiv, subdiv) {
    }

    grid_iterator(unsigned lx, unsigned ly, unsigned lz)
        : x(0), y(0), z(0), lx(lx), ly(ly), lz(lz) {
    }

    bool over() const {
        return x >= lx;
    }

    std::tuple<unsigned, unsigned, unsigned> operator*() const {
        return { x, y, z };
    }

    grid_iterator& operator++() {
        if (!over()) {
            z++;
            if (z == lz) {
                z = 0;
                y++;
                if (y == ly) {
                    y = 0;
                    x++;
                }
            }
        } else {
            return *this;
        }
        return *this;
    }
};

mesh marching_cubes::generate(std::vector<metaball> const &metaballs, params const &params) {
    mesh ret;

    glm::vec3 bMin, bMax;
    calc_bounds(bMin, bMax, metaballs);

    grid3d G;
    float sx = bMax.x - bMin.x;
    float sy = bMax.y - bMin.y;
    float sz = bMax.z - bMin.z;
    G.set_grid_dimensions(params.subdivisions, params.subdivisions, params.subdivisions);

    float cx  = params.subdivisions / (float)sx;
    float cy  = params.subdivisions / (float)sy;
    float cz  = params.subdivisions / (float)sz;

    float icx = sx / (float)params.subdivisions;
    float icy = sy / (float)params.subdivisions;
    float icz = sz / (float)params.subdivisions;

    for (auto it = grid_iterator(params.subdivisions); !it.over(); ++it) {
        float min_dist = FLT_MAX;
        auto pos = glm::vec3(it.x * icx, it.y * icy, it.z * icz) + bMin;
        for (auto &mb : metaballs) {
            float dist = length(pos - mb.position) - mb.radius;
            if (dist < min_dist) {
                min_dist = dist;
            }
        }

        G.set_grid_value(it.x, it.y, it.z, min_dist);
    }

    MC33 M;
    M.set_grid3d(&G);
    auto surf = M.calculate_isosurface(0.0f);
    auto vN = surf->get_num_vertices();
    for (unsigned i = 0; i < vN; i++) {
        auto pos = surf->getVertex(i);
        auto normal = surf->getNormal(i);
        ret.positions.push_back(glm::vec3(pos[0] * icx, pos[1] * icy, pos[2] * icz) + bMin);
        ret.normal.push_back({ -normal[0], -normal[1], -normal[2] });
        ret.uv.push_back({ 0, 0 });
    }
    auto tN = surf->get_num_triangles();
    for (unsigned i = 0; i < tN; i++) {
        auto tri = surf->getTriangle(i);
        for (int v = 0; v < 3; v++) {
            ret.indices.push_back(tri[v]);
        }
    }
    delete surf;

    return ret;
}