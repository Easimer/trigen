// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: triangle-based mesh builder
//

#include <unordered_set>
#include <array>
#include <vector>
#include <optional>
#include <trigen/linear_math.h>

class Mesh_Builder {
public:
    using Vertex = std::array<float, 3>;
    static_assert(sizeof(Vertex) == 3 * sizeof(float), "!!!");

    struct Triangle {
        std::array<Vertex, 3> vertices;
    };

    static_assert(sizeof(Triangle) == 3 * 3 * sizeof(float), "!!!");

    struct Optimized_Mesh {
        std::vector<Vertex> vertices;
        std::vector<unsigned int> elements;

        size_t VerticesSize() const {
            return vertices.size() * sizeof(Vertex);
        }

        size_t ElementsSize() const {
            return elements.size() * sizeof(unsigned int);
        }
    };

    void PushTriangle(lm::Vector4 const& v0, lm::Vector4 const& v1, lm::Vector4 const& v2) {
        std::array<Vertex, 3> vertices = {
            Vertex {v0[0], v0[1], v0[2]},
            Vertex {v1[0], v1[1], v1[2]},
            Vertex {v2[0], v2[1], v2[2]},
        };
        triangles.push_back({ vertices });
    }

    float Metric(Vertex const& lhs, Vertex const& rhs) const {
        auto dx = lhs[0] - rhs[0];
        auto dy = lhs[1] - rhs[1];
        auto dz = lhs[2] - rhs[2];
        return dx * dx + dy * dy + dz * dz;
    }

    const float Epsilon = 0.1f;

    Optimized_Mesh Optimize() const {
        Optimized_Mesh ret;
        auto orig = 3 * triangles.size();

        // For every triangle in this mesh
        for (auto const& tri : triangles) {
            // For every vertex in the current triangle
            for (auto vtx = 0; vtx < 3; vtx++) {
                // Try to find the element index if exists
                std::optional<unsigned int> elementIndex;
                auto const& v = tri.vertices[vtx];

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
        printf("Mesh_Builder::Optimize: reduction: %fx\n", reduction);

        return ret;
    }

private:
    std::vector<Triangle> triangles;
};
