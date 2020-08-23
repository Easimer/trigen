// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#include <trigen/tree_meshifier.h>
#include <trigen/catmull_rom.h>
#include <trigen/meshbuilder.h>
#include <trigen/future_union_mesh.h>
#include "trunk_generator.h"

Optimized_Mesh<TG_Vertex> ProcessNodes(Tree_Node_Pool const& tree, uint32_t const uiStart, uint32_t const uiBranch, uint32_t const uiEnd, TM_RadiusFunc const& radius_func) {
    std::vector<lm::Vector4> points;
    std::vector<uint64_t> user_data;

    // start node might have multiple children.
    // the code below won't handle that properly, so we add
    // that node here
    auto const& pStart = tree.GetNode(uiStart);
    auto const& pSecond = tree.GetNode(uiBranch);
    points.push_back(pStart.vPosition - (pSecond.vPosition - pStart.vPosition));
    user_data.push_back(pStart.unUser);
    points.push_back(pStart.vPosition);
    user_data.push_back(pStart.unUser);
    uint32_t uiCursor = uiBranch;
    uint32_t uiPrev = uiStart;
    while(1) {
        auto const& cur = tree.GetNode(uiCursor);
        points.push_back(cur.vPosition);
        user_data.push_back(cur.unUser);

        if (uiCursor == uiEnd) {
            break;
        }

        assert(cur.unChildCount == 1);
        uiPrev = uiCursor;
        uiCursor = cur.aiChildren[0];
    };
    auto const& pEnd = tree.GetNode(uiEnd);
    auto const& pPenultimate = tree.GetNode(uiPrev);
    points.push_back(pEnd.vPosition + (pEnd.vPosition - pPenultimate.vPosition));
    user_data.push_back(pPenultimate.unUser);

    /*
    printf("(%f, %f, %f) -> (%f, %f, %f)\n",
        pStart.vPosition[0], pStart.vPosition[1], pStart.vPosition[2],
        pEnd.vPosition[0], pEnd.vPosition[1], pEnd.vPosition[2]
        );
    */

    Catmull_Rom_Composite<lm::Vector4> cr(points.size(), points.data(), user_data.data());
    return MeshFromSpline(cr, radius_func);
}

static Future_Union_Mesh<TG_Vertex> ProcessMultiNode(Tree_Node_Pool const& tree, uint32_t const uiNode, TM_RadiusFunc const& radius_func) {
    Future_Union_Mesh<TG_Vertex> ret;

    auto const& node = tree.GetNode(uiNode);
    // assert(node.unChildCount > 1);
    for (uint32_t uiChildOff = 0; uiChildOff < node.unChildCount; uiChildOff++) {
        auto const uiBranchHead = node.aiChildren[uiChildOff];
        auto uiCurrent = uiBranchHead;
        auto const* pCurrent = &tree.GetNode(uiCurrent);
        while (pCurrent->unChildCount == 1) {
            uiCurrent = pCurrent->aiChildren[0];
            pCurrent = &tree.GetNode(uiCurrent);
        }

        Union(ret, std::async(std::launch::async, &ProcessNodes, tree, uiNode, uiBranchHead, uiCurrent, radius_func));
        if (pCurrent->unChildCount > 1) {
            Union(ret, ProcessMultiNode(tree, uiCurrent, radius_func));
        }
    }

    return ret;
}

Optimized_Mesh<TG_Vertex> ProcessTree(Tree_Node_Pool const& tree, TM_RadiusFunc const& radius_func) {
    auto const& root = tree.GetNode(0);
    assert(root.unChildCount > 0);

    return (Optimized_Mesh<TG_Vertex>)ProcessMultiNode(tree, 0, radius_func);
}

