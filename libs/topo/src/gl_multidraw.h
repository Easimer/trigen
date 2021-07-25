// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <functional>
#include <unordered_map>

#include "copy_workers.h"
#include "gl_model_manager.h"
#include "gl_utils.h"
#include "glres.h"
#include "material_manager.h"
#include "render_queue.h"
#include "renderable_manager.h"

#include <glad/glad.h>

namespace topo {
class Shader_Model_Matrix_Compute {
public:
    void
    Build();

    gl::Shader_Program &
    Program() {
        return _program;
    }

    gl::Uniform_Location<GLuint> &
    TotalItemCount() {
        return _locTotalItemCount;
    }

private:
    gl::Shader_Program _program;
    gl::Uniform_Location<GLuint> _locTotalItemCount;
};

class GL_Multidraw {
public:
    GL_Multidraw(
        Render_Queue *rq,
        Renderable_Manager *renderableManager,
        Material_Manager *materialManager,
        GL_Model_Manager *modelManager,
        Shader_Model_Matrix_Compute *shaderModelMatrixCompute,
        GLint maxShaderStorageBlockSize,
        Copy_Workers *copyWorkers);

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

        GLuint lastBatchSize;
        Vector<GLuint> indirectBuffers;

        Vector<GLuint> modelMatrixIndices;
        Vector<GLuint> modelMatrixIndexBuffers;
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
    using Draw_Data = Map<Material_Type, Map<Material_ID, Material_Instance>>;
    struct Model_Matrix_Info {
        std::vector<glm::vec4> translate;
        std::vector<glm::mat4> rotateMatrix;
        std::vector<glm::vec4> scale;
    };

    struct Managers {
        Renderable_Manager *renderable;
        Material_Manager *material;
        GL_Model_Manager *model;
    };

    void
    MakeDrawData(
        Render_Queue *rq,
        Managers const &managers,
        Model_Matrix_Info &mmInfo);

    void
    ComputeModelMatrices(Model_Matrix_Info const &mmInfo);

    size_t _batchSize = 256;
    GLuint _modelMatrixBuffer;
    Map<Material_Type, Map<Material_ID, Material_Instance>> _drawData;
    Shader_Model_Matrix_Compute *_shaderModelMatrixCompute;
    Array_Recycler<gl::VBO> *_vboRecycler;
    Copy_Workers *_copyWorkers;
};
}
