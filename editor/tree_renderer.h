// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#pragma once

#include "render_queue.h"
#include <trigen/tree_meshifier.h>

namespace Tree_Renderer {
    struct _Tree;
    // Tree handle
    using Tree = _Tree*;

    class ITree_Source {
    public:
        virtual Tree_Node_Pool const* GetNodes() = 0;
        virtual bool DidTreeChange() = 0;
    };

    Tree CreateTree(ITree_Source* pSource);
    void DestroyTree(Tree hTree);
    void Render(rq::Render_Queue* rq, Tree hTree, lm::Vector4 const& vPosition, gl::Shader_Program const& program);
}


