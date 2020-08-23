// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#pragma once

#include <cstdint>
#include <cassert>
#include <vector>
#include <functional>
#include <trigen/linear_math.h>
#include <trigen/meshbuilder.h>

constexpr uint32_t gunTreeNodeMaxChildren = 11;

struct Tree_Node {
    // Node's position in the world
    lm::Vector4 vPosition;
    // User data
    uint64_t unUser;
    // Array of child nodes
    uint32_t aiChildren[gunTreeNodeMaxChildren];
    // Number of child nodes in the array above
    uint32_t unChildCount = 0;

    bool AddChild(uint32_t uiIndex) {
        bool ret = false;

        if (unChildCount < gunTreeNodeMaxChildren) {
            aiChildren[unChildCount] = uiIndex;
            unChildCount++;
        }

        return ret;
    }
};

class Tree_Node_Pool {
public:
    Tree_Node& GetNode(uint32_t uiIndex) {
        assert(uiIndex < pool.size());
        return pool[uiIndex];
    }

    Tree_Node const& GetNode(uint32_t uiIndex) const {
        assert(uiIndex < pool.size());
        return pool[uiIndex];
    }

    Tree_Node& Allocate(uint32_t& uiIndex) {
        uiIndex = pool.size();
        pool.push_back(Tree_Node());
        return pool.back();
    }
private:
    std::vector<Tree_Node> pool;
};

using TM_RadiusFunc = std::function<float(size_t iIndex, lm::Vector4 const& vPosition, uint64_t user0, float weight0, uint64_t user1, float weight1)>;

struct TG_Vertex {
    std::array<float, 4> position;
    std::array<float, 2> uv;

    float metric(TG_Vertex const& other) const {
        auto dx = other.position[0] - position[0];
        auto dy = other.position[1] - position[1];
        auto dz = other.position[2] - position[2];

        return dx * dx + dy * dy + dz * dz;
    }
};

Optimized_Mesh<TG_Vertex> ProcessTree(Tree_Node_Pool const& tree, TM_RadiusFunc const& radius_func);
