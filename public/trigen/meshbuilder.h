// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: triangle-based mesh builder
//

#pragma once
#include <unordered_set>
#include <array>
#include <vector>
#include <optional>
#include <trigen/linear_math.h>

#include <trigen/profiler.h>

struct ExampleVertex {
    std::array<float, 3> position;
    std::array<float, 2> uv;
    float triangles_favorite_number;

    float metric(ExampleVertex const& other) const {
        auto dx = other.position[0] - position[0];
        auto dy = other.position[1] - position[1];
        auto dz = other.position[2] - position[2];

        return dx * dx + dy * dy + dz * dz;
    }
};

template<typename Vertex>
struct Optimized_Mesh {
    static_assert(std::is_standard_layout_v<Vertex>, "Vertex type must have a standard layout!");

    std::vector<Vertex> vertices;
    std::vector<unsigned int> elements;

    size_t VerticesSize() const {
        return vertices.size() * sizeof(Vertex);
    }

    size_t ElementsSize() const {
        return elements.size() * sizeof(unsigned int);
    }

    size_t VerticesCount() const {
        return vertices.size();
    }

    size_t ElementsCount() const {
        return elements.size();
    }
};

template<typename Vertex>
class Mesh_Builder {
public:
    static_assert(std::is_standard_layout_v<Vertex>, "Vertex type must have a standard layout!");

    using Triangle = std::array<Vertex, 3>;

    Mesh_Builder(float flEpsilon = 0.1f) : Epsilon(flEpsilon), triangles{} {
        triangles.reserve(24);
    }

    void PushTriangle(Vertex const& v0, Vertex const& v1, Vertex const& v2) {
        triangles.push_back({ v0, v1, v2 });
    }

    static float Metric(Vertex const& lhs, Vertex const& rhs) {
        return lhs.metric(rhs);
    }

    // TODO(danielm): calculate this value when Optimize is called
    const float Epsilon;

    Optimized_Mesh<Vertex> Optimize() const {
        SCOPE_BENCHMARK();
        Optimized_Mesh<Vertex> ret;
        auto orig = 3 * triangles.size();

        // For every triangle in this mesh
        for (auto const& tri : triangles) {
            // For every vertex in the current triangle
            for (auto vtx = 0; vtx < 3; vtx++) {
                // Try to find the element index if exists
                std::optional<unsigned int> elementIndex;
                auto const& v = tri[vtx];

                for (unsigned int i = 0; i < ret.vertices.size() && !elementIndex; i++) {
                    // If one of the vertices is at least "good enough" then use it
                    auto M = Metric(v, ret.vertices[i]);
                    if (M <= Epsilon) {
                        elementIndex = i;
                    }
                }

                // Couldn't find any good vertex
                if (!elementIndex) {
                    // Push the new vertex into the vector
                    elementIndex = ret.vertices.size();
                    ret.vertices.push_back(v);
                }

                ret.elements.push_back(elementIndex.value());
            }
        }

        auto res = ret.vertices.size();

        auto reduction = res / (float)orig;
        // printf("Mesh_Builder::Optimize: reduction: %fx\n", reduction);

        return ret;
    }

private:
    std::vector<Triangle> triangles;
};

template<typename Vertex>
inline Optimized_Mesh<Vertex> UnionGlobal(
    Optimized_Mesh<Vertex> const& lhs,
    Optimized_Mesh<Vertex> const& rhs) {
    SCOPE_BENCHMARK();
    Optimized_Mesh<Vertex> ret;
    auto const flEpsilon = 0.00001f;
    std::array<Optimized_Mesh<Vertex> const*, 2> aMeshes = { &lhs, &rhs };

    for (auto const& mesh : aMeshes) {
        for (size_t iElem = 0; iElem < mesh->elements.size(); iElem ++) {
            std::optional<unsigned int> elementIndex;
            auto iVtx = mesh->elements[iElem];
            for (size_t iOptVtx = 0; iOptVtx < ret.vertices.size() && !elementIndex; iOptVtx++) {
                auto M = Mesh_Builder<Vertex>::Metric(mesh->vertices[iVtx], ret.vertices[iOptVtx]);
                if (M <= flEpsilon) {
                    elementIndex = iOptVtx;
                }
            }

            if (!elementIndex) {
                elementIndex = ret.vertices.size();
                ret.vertices.push_back(mesh->vertices[iVtx]);
            }

            ret.elements.push_back(elementIndex.value());
        }
    }

    return ret;
}

template<typename Vertex>
inline Optimized_Mesh<Vertex> UnionWindow(
    Optimized_Mesh<Vertex> const& lhs,
    Optimized_Mesh<Vertex> const& rhs,
    size_t unWindowSize) {
    SCOPE_BENCHMARK();
    Optimized_Mesh<Vertex> ret;
    auto const flEpsilon = 0.00001f;
    std::array<Optimized_Mesh<Vertex> const*, 2> aMeshes = { &lhs, &rhs };

    for (auto const& mesh : aMeshes) {
        for (size_t iElem = 0; iElem < mesh->elements.size(); iElem ++) {
            std::optional<unsigned int> elementIndex;
            auto iVtx = mesh->elements[iElem];
            auto const uiStart = ret.vertices.size() > unWindowSize ? ret.vertices.size() - unWindowSize : 0;
            for (size_t iOptVtx = uiStart; iOptVtx < ret.vertices.size() && !elementIndex; iOptVtx++) {
                auto M = Mesh_Builder<Vertex>::Metric(mesh->vertices[iVtx], ret.vertices[iOptVtx]);
                if (M <= flEpsilon) {
                    elementIndex = iOptVtx;
                }
            }

            if (!elementIndex) {
                elementIndex = ret.vertices.size();
                ret.vertices.push_back(mesh->vertices[iVtx]);
            }

            ret.elements.push_back(elementIndex.value());
        }
    }

    return ret;
}

// Computes the union of two optimized meshes
template<typename Vertex>
inline Optimized_Mesh<Vertex> operator+(
    Optimized_Mesh<Vertex> const& lhs,
    Optimized_Mesh<Vertex> const& rhs) {
    auto const unPointCount = lhs.vertices.size() + rhs.vertices.size();
    if (unPointCount > 256) {
        return UnionWindow<Vertex>(lhs, rhs, 32);
    } else {
        return UnionGlobal<Vertex>(lhs, rhs);
    }
}
