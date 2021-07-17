// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <functional>
#include <unordered_map>

#include "gl_model_manager.h"
#include "material_manager.h"
#include "render_queue.h"
#include "renderable_manager.h"

#include <glad/glad.h>

namespace topo {
class GL_Multidraw {
public:
    GL_Multidraw(
        Render_Queue *rq,
        Renderable_Manager *renderableManager,
        Material_Manager *materialManager,
        GL_Model_Manager *modelManager);

    ~GL_Multidraw();

    struct GL_Indirect {
        GLuint count;
        GLuint instanceCount;
        GLuint firstIndex;
        GLuint baseVertex;
        GLuint baseInstance;
    };

    template<typename K, typename V>
    using Map = std::unordered_map<K, V>;
    template<typename T>
    using Vector = std::vector<T>;

    struct Instances {
        Vector<GL_Indirect> indirects;
        Vector<glm::mat4> modelMatrices;

        GLuint lastBatchSize;
        Vector<GLuint> indirectBuffers;
        Vector<GLuint> modelMatrixBuffers;
    };

    struct Material_Instance {
        Map<GLenum, Instances> instances;
    };

    using Shader_Setup_Callback = std::function<void(topo::Material_Type materialType)>;
    using Material_Setup_Callback = std::function<void(topo::Material_Type materialType, topo::Material_ID material)>;

    void
    Execute(
        Shader_Setup_Callback const &setupShader,
        Material_Setup_Callback const &setupMaterial);

private:
    size_t _batchSize = 256;
    Map<Material_Type, Map<Material_ID, Material_Instance>> _drawData;
};
}
