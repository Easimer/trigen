// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#include "uv_quadtree.h"

#include <memory>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/polygon.hpp>

#include <boost/geometry/index/rtree.hpp>

#include <Tracy.hpp>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

using Point = bg::model::point<float, 2, bg::cs::cartesian>;
using Box = bg::model::box<Point>;
using Polygon = bg::model::polygon<Point, false, true>;
using Value = std::pair<Box, size_t>;

using RTree = bgi::rtree<Value, bgi::quadratic<16>>;

static bool is_same_side(glm::vec3 p1, glm::vec3 p2, glm::vec3 a, glm::vec3 b) {
    auto cp1 = cross(b - a, p1 - a);
    auto cp2 = cross(b - a, p2 - a);
    return dot(cp1, cp2) >= 0;
}

static bool is_point_in_triangle(glm::vec2 x, glm::vec2 p0, glm::vec2 p1, glm::vec2 p2) {
    auto a = glm::vec3(p0, 0.f);
    auto b = glm::vec3(p1, 0.f);
    auto c = glm::vec3(p2, 0.f);
    auto p = glm::vec3(x, 0.f);
    return (is_same_side(p, a, b, c) && is_same_side(p, b, a, c) && is_same_side(p, c, a, b));
}

class UVSpatialIndex_Impl : public UVSpatialIndex {
public:
    ~UVSpatialIndex_Impl() override = default;

    UVSpatialIndex_Impl(PSP::Mesh const *mesh, RTree &&tree) : _mesh(mesh), _tree(std::move(tree)) {
    }

    std::optional<size_t> find_triangle(glm::vec2 const &uv) override {
        ZoneScoped;
        std::vector<Value> results;

        auto p = Point(uv.x, uv.y);
        _tree.query(bgi::contains(p), std::back_inserter(results));

        for (auto& value : results) {
            auto t = value.second;

            auto uv0 = _mesh->uv[_mesh->elements[t * 3 + 0]];
            auto uv1 = _mesh->uv[_mesh->elements[t * 3 + 1]];
            auto uv2 = _mesh->uv[_mesh->elements[t * 3 + 2]];

            auto area = 0.5f * glm::abs(glm::determinant(glm::mat3(glm::vec3(uv0, 1), glm::vec3(uv1, 1), glm::vec3(uv2, 1))));

            if (area > 0 && is_point_in_triangle(uv, uv0, uv1, uv2)) {
                return t;
            }
        }

        return std::nullopt;
    }

private:
    PSP::Mesh const *_mesh;
    RTree _tree;
};

std::unique_ptr<UVSpatialIndex> make_uv_spatial_index(PSP::Mesh const *mesh) {
    ZoneScoped;
    std::vector<std::pair<Polygon, size_t>> polygons;

    auto N = mesh->elements.size() / 3;
    for (size_t t = 0; t < N; t++) {
        Polygon p;
        for (int i = 0; i < 3; i++) {
            auto uv = mesh->uv[mesh->elements[t * 3 + i]];
            p.outer().push_back(Point(uv.x, uv.y));
        }

        polygons.push_back(std::make_pair(p, t));
    }

    bgi::rtree<Value, bgi::quadratic<16>> rtree;

    for (auto &poly : polygons) {
        auto boundingBox = bg::return_envelope<Box>(poly.first);
        rtree.insert(std::make_pair(boundingBox, poly.second));
    }

    return std::make_unique<UVSpatialIndex_Impl>(mesh, std::move(rtree));
}
