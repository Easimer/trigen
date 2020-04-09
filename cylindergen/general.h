// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: General data structures
//

#pragma once
#include <trigen/linear_math.h>
#include <cstdint>
#include <cassert>
#include <vector>

constexpr uint32_t gunTreeNodeMaxChildren = 11;

struct Tree_Node {
    lm::Vector4 vPosition;
    uint32_t aiChildren[gunTreeNodeMaxChildren];
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
    Tree_Node_Pool() = default;

    Tree_Node_Pool(Tree_Node const&) {
        // TODO(danielm): copy the other tree
    }

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
