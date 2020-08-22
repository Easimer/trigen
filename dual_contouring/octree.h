// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: octree
//

#pragma once

#include <array>
#include <memory>
#include <glm/common.hpp>
#include <glm/vec3.hpp>
#include <glm/exponential.hpp>

template<int MaxLevels>
class Octree {
    static_assert(MaxLevels > 0, "MaxLevels must be greater than 0!");
public:
    Octree(glm::vec3 const& center, glm::vec3 const& extent) : root{glm::vec4(center, 0), {}}, extent(extent) {}

    void add_point(glm::vec3 const& p) {
        add_point(0, root, glm::vec4(p, 0));
    }

    bool count(glm::vec3 const& p) const {
        int level = 0;
        Node const* node = &root;

        while (level != MaxLevels) {
            auto delta = glm::vec4(p, 0) - node->center;
            auto octant = DirToOctant(delta);
            assert(octant >= 0 && octant < Octant::Max);

            auto child_idx = (unsigned)octant;
            auto& child = node->children[child_idx];

            if (!child) {
                return false;
            }

            node = child.get();

            level++;
        }

        assert(node != NULL);

        return true;
    }

private:
    enum class Octant : size_t {
        Top_Front_Left = 0,
        Top_Front_Right,
        Top_Back_Left,
        Top_Back_Right,
        Bottom_Front_Left,
        Bottom_Front_Right,
        Bottom_Back_Left,
        Bottom_Back_Right,

        Max,
    };

    struct Node {
        glm::vec4 center;
        std::array<std::unique_ptr<Node>, (size_t)Octant::Max> children;
    };

    void add_point(int level, Node& node, glm::vec4 const& p) {
        if (level != MaxLevels) {
            auto delta = p - node.center;
            auto octant = DirToOctant(delta);
            assert(octant >= 0 && octant < Octant::Max);

            auto child_idx = (unsigned)octant;

            auto& child = node.children[child_idx];
            node.children[child_idx] = std::make_unique<Node>(Node({ get_new_node_center(level + 1, node.center, delta), {} }));

            assert(child);

            if (child) {
                add_point(level + 1, *child, p);
            }
        }
    }

    Octant DirToOctant(glm::vec4 const& delta) const {
        if (delta.x >= 0) {
            if (delta.y >= 0) {
                if (delta.z >= 0) {
                    return Octant::Top_Front_Right;
                } else {
                    return Octant::Top_Back_Right;
                }
            } else {
                if (delta.z >= 0) {
                    return Octant::Bottom_Front_Right;
                } else {
                    return Octant::Bottom_Back_Right;
                }
            }
        } else {
            if (delta.y >= 0) {
                if (delta.z >= 0) {
                    return Octant::Top_Front_Left;
                } else {
                    return Octant::Top_Back_Left;
                }
            } else {
                if (delta.z >= 0) {
                    return Octant::Bottom_Front_Left;
                } else {
                    return Octant::Bottom_Back_Left;
                }
            }
        }
    }

    glm::vec4 get_new_node_center(int level, glm::vec4 const& parent_center, glm::vec4 delta) const {
        auto step = glm::pow(0.5f, level);

        delta.x /= glm::abs(delta.x);
        delta.y /= glm::abs(delta.y);
        delta.z /= glm::abs(delta.z);

        return parent_center + step * delta;
    }

    Node root;
    glm::vec3 extent;
};
