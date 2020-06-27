// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#include "stdafx.h"
#include "tree_renderer.h"
#include <cassert>
#include <trigen/tree_meshifier.h>
#include <trigen/glres.h>
#include <trigen/gl.h>

namespace Tree_Renderer {
    struct _Tree {
        ITree_Source* pSource;
        gl::VAO vao;
        gl::VBO vbo_vertices, vbo_elements;
        size_t element_count;
    };

    Tree CreateTree(ITree_Source* pSource) {
        Tree ret = NULL;

        if (pSource != NULL) {
            ret = new _Tree;
            ret->pSource = pSource;
        }

        return ret;
    }

    void DestroyTree(Tree hTree) {
        if (hTree != NULL) {
            delete hTree;
        }
    }

    void Render(rq::Render_Queue* rq, Tree hTree, lm::Vector4 const& vPosition, gl::Shader_Program const& program) {
        if (hTree != NULL && rq != NULL) {
            assert(hTree->pSource != NULL);
            if (hTree->pSource->DidTreeChange()) {
                auto const pNodePool = hTree->pSource->GetNodes();
                if (pNodePool != NULL) {
                    auto const& nodePool = *hTree->pSource->GetNodes();
                    auto mesh = ProcessTree(nodePool);
                    gl::Bind(hTree->vao);
                    gl::UploadArray(hTree->vbo_vertices, mesh.VerticesSize(), mesh.vertices.data());
                    gl::UploadElementArray(hTree->vbo_elements, mesh.ElementsSize(), mesh.elements.data());
                    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
                    glEnableVertexAttribArray(0);
                    hTree->element_count = mesh.elements.size();
                }
            }
            rq::Render_Command rc;

            rc.kind = rq::k_unRCChangeProgram;
            rc.change_program = rq::MakeChangeProgramParams(program);
            rq->Add(rc);

            rc.kind = rq::k_uNRCDrawElementModel;
            rc.draw_element_model = rq::MakeDrawElementModelParams(hTree->vao, hTree->element_count, vPosition);
            rq->Add(rc);
        }
    }
}
