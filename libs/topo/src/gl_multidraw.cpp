// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "gl_multidraw.h"

#include "gl_shadercompiler.h"
#include "shader_program_builder.h"

#include <Tracy.hpp>

extern "C" {
extern char const *model_matrix_compute_glsl;
}

enum {
    MODEL_MATRIX_COMPUTE_BINDING_START = 0,
    MODEL_MATRIX_COMPUTE_BINDING_TRANSLATE = MODEL_MATRIX_COMPUTE_BINDING_START,
    MODEL_MATRIX_COMPUTE_BINDING_ROTATE,
    MODEL_MATRIX_COMPUTE_BINDING_SCALE,
    MODEL_MATRIX_COMPUTE_BINDING_INPUT_MAX,
    MODEL_MATRIX_COMPUTE_BINDING_OUTPUT = MODEL_MATRIX_COMPUTE_BINDING_INPUT_MAX
};

namespace topo {

GL_Multidraw::GL_Multidraw(
    Render_Queue *rq,
    Renderable_Manager *renderableManager,
    Material_Manager *materialManager,
    GL_Model_Manager *modelManager,
    Shader_Model_Matrix_Compute *shaderModelMatrixCompute)
    : _shaderModelMatrixCompute(shaderModelMatrixCompute) {
    ZoneScoped;
    std::vector<Render_Queue::Command> commands;

    GLint maxShaderStorageBlockSize;
    glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &maxShaderStorageBlockSize);

    _batchSize = maxShaderStorageBlockSize / sizeof(glm::mat4);

    for (auto &cmd : rq->GetCommands()) {
        if (renderableManager->GetRenderableKind(cmd.renderable)
            == Renderable_Manager::RENDERABLE_MODEL) {
            Model_ID model;
            Material_ID material;
            renderableManager->GetModelAndMaterial(
                cmd.renderable, &model, &material);

            auto materialKind = materialManager->GetType(material);

            if (_drawData.count(materialKind) == 0) {
                _drawData[materialKind] = {};
            }
            auto &materialInstances = _drawData[materialKind];

            if (materialInstances.count(material) == 0) {
                materialInstances[material] = {};
            }
            auto &instancesByIndexType = materialInstances[material].instances;

            GLuint indexOffset;
            GLuint firstIndex;
            GLuint baseVertex;
            GLenum elementType;
            size_t numElements;

            modelManager->GetDrawParameters(
                model, &indexOffset, &baseVertex, &elementType, &numElements,
                &firstIndex);

            if (instancesByIndexType.count(elementType) == 0) {
                instancesByIndexType[elementType] = {};
            }
            auto &instances = instancesByIndexType[elementType];

            GL_Indirect indirect;
            indirect.baseInstance = 0;
            indirect.instanceCount = 1;
            indirect.firstIndex = firstIndex;
            indirect.count = numElements;
            indirect.baseVertex = baseVertex;

            instances.modelTranslateMatrices.emplace_back(
                glm::translate(cmd.transform.position));
            instances.modelRotateMatrices.emplace_back(
                glm::mat4_cast(cmd.transform.rotation));
            instances.modelScaleMatrices.emplace_back(
                glm::scale(cmd.transform.scale));
            instances.indirects.push_back(indirect);
        }
    }

    // Calculate the multiplied model matrices on the GPU
    glUseProgram(_shaderModelMatrixCompute->Program());
    for (auto &materialGroup : _drawData) {
        for (auto &materialInstance : materialGroup.second) {
            for (auto &indexType : materialInstance.second.instances) {
                auto &instances = indexType.second;
                size_t remain = instances.modelTranslateMatrices.size();
                size_t next = 0;
                unsigned currentBatchSize = 0;

                std::vector<glm::mat4>
                    *matrixVectors[MODEL_MATRIX_COMPUTE_BINDING_INPUT_MAX]
                    = { &instances.modelTranslateMatrices,
                        &instances.modelRotateMatrices,
                        &instances.modelScaleMatrices };

                while (remain > 0) {
                    currentBatchSize
                        = (remain > _batchSize) ? _batchSize : remain;
                    gl::VBO
                        inputBuffers[MODEL_MATRIX_COMPUTE_BINDING_INPUT_MAX];
                    GLuint bufModelMatrices;

                    // Create SSBO where the final matrices will be stored
                    glGenBuffers(1, &bufModelMatrices);
                    glBindBuffer(
                        GL_SHADER_STORAGE_BUFFER, bufModelMatrices);
                    glBufferStorage(
                        GL_SHADER_STORAGE_BUFFER,
                        currentBatchSize * sizeof(glm::mat4), nullptr, 0);

                    // Create SSBOs for the inputs
                    for (int i = MODEL_MATRIX_COMPUTE_BINDING_START;
                         i < MODEL_MATRIX_COMPUTE_BINDING_INPUT_MAX; i++) {
                        glBindBuffer(GL_SHADER_STORAGE_BUFFER, inputBuffers[i]);
                        glBufferStorage(
                            GL_SHADER_STORAGE_BUFFER,
                            currentBatchSize * sizeof(glm::mat4),
                            matrixVectors[i]->data() + next, 0);
                    }

                    gl::SetUniformLocation(
                        _shaderModelMatrixCompute->TotalItemCount(),
                        currentBatchSize);

                    for (int i = MODEL_MATRIX_COMPUTE_BINDING_START;
                         i < MODEL_MATRIX_COMPUTE_BINDING_INPUT_MAX; i++) {
                        glBindBufferBase(
                            GL_SHADER_STORAGE_BUFFER, i, inputBuffers[i]);
                    }
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, bufModelMatrices);

                    auto const numItems = 16;
                    auto numGroups = (currentBatchSize + numItems - 1) / numItems;
                    glDispatchComputeGroupSizeARB(numGroups, 1, 1, numItems, 1, 1);

                    instances.modelMatrixBuffers.emplace_back(bufModelMatrices);

                    remain -= currentBatchSize;
                    next += currentBatchSize;
                }
            }
        }
    }

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // Upload the draw indirect and uniform buffers
    for (auto &materialGroup : _drawData) {
        for (auto& materialInstance : materialGroup.second) {
            for (auto &indexType : materialInstance.second.instances) {
                size_t remain = indexType.second.indirects.size();
                size_t next = 0;
                size_t currentBatchSize = 0;
                auto &indirects = indexType.second.indirects;

                while (remain > 0) {
                    currentBatchSize
                        = (remain > _batchSize) ? _batchSize : remain;

                    GLuint bufIndirect;
                    glGenBuffers(1, &bufIndirect);

                    auto flags = GL_MAP_WRITE_BIT;

                    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, bufIndirect);
                    auto size = currentBatchSize * sizeof(GL_Indirect);
                    glBufferStorage(
                        GL_DRAW_INDIRECT_BUFFER, size, nullptr, flags);
                    auto indirectsDevice = glMapBufferRange(
                        GL_DRAW_INDIRECT_BUFFER, 0, size, flags);
                    auto indirectsHost = indirects.data() + next;
                    memcpy(indirectsDevice, indirectsHost, size);
                    glUnmapBuffer(GL_DRAW_INDIRECT_BUFFER);

                    indexType.second.indirectBuffers.emplace_back(
                        std::move(bufIndirect));

                    remain -= currentBatchSize;
                    next += currentBatchSize;
                }

                indexType.second.lastBatchSize = currentBatchSize;
            }
        }
    }
}

GL_Multidraw::~GL_Multidraw() {
    ZoneScoped;
    for (auto &materialGroup : _drawData) {
        for (auto &materialInstance : materialGroup.second) {
            for (auto &indexType : materialInstance.second.instances) {
                glDeleteBuffers(
                    indexType.second.indirectBuffers.size(),
                    indexType.second.indirectBuffers.data());
                glDeleteBuffers(
                    indexType.second.modelMatrixBuffers.size(),
                    indexType.second.modelMatrixBuffers.data());
            }
        }
    }
}

void
GL_Multidraw::Execute(
    Shader_Setup_Callback const &setupShader,
    Material_Setup_Callback const &setupMaterial) {
    ZoneScoped;

    for (auto &materialGroup : _drawData) {
        setupShader(materialGroup.first);
        for (auto &materialInstance : materialGroup.second) {
            setupMaterial(materialGroup.first, materialInstance.first);
            for (auto &indexType : materialInstance.second.instances) {
                auto N = indexType.second.indirectBuffers.size();

                for (size_t i = 0; i < N; i++) {
                    auto &bufIndirect = indexType.second.indirectBuffers[i];
                    auto &bufUniformMatrices = indexType.second.modelMatrixBuffers[i];
                    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, bufIndirect);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufUniformMatrices);
                    auto batchSize = (i == N - 1)
                        ? indexType.second.lastBatchSize
                        : _batchSize;
                    glMultiDrawElementsIndirect(
                        GL_TRIANGLES, indexType.first, nullptr, batchSize, 0);
                }
            }
        }
    }
}

void
Shader_Model_Matrix_Compute::Build() {
    printf("[ topo ] building shader 'model matrix compute'\n");
    auto defines = GL_Shader_Define_List {
        { "BINDING_TRANSLATE",
          std::to_string(MODEL_MATRIX_COMPUTE_BINDING_TRANSLATE) },
        { "BINDING_ROTATE",
          std::to_string(MODEL_MATRIX_COMPUTE_BINDING_ROTATE) },
        { "BINDING_SCALE", std::to_string(MODEL_MATRIX_COMPUTE_BINDING_SCALE) },
        { "BINDING_OUTPUT",
          std::to_string(MODEL_MATRIX_COMPUTE_BINDING_OUTPUT) }
    };
    auto csh = FromStringLoadShader<GL_COMPUTE_SHADER>(model_matrix_compute_glsl, defines);

    auto builder = gl::Shader_Program_Builder();
    auto program = builder.Attach(csh).Link();
    if (program) {
        _program = std::move(program.value());

        _locTotalItemCount = gl::Uniform_Location<GLuint>(_program, "uiTotalItemCount");
    } else {
        throw Shader_Linker_Exception("textured lit", builder.Error());
    }

}

}
