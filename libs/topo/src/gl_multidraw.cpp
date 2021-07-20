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
    Shader_Model_Matrix_Compute *shaderModelMatrixCompute,
    GLint maxShaderStorageBlockSize)
    : _shaderModelMatrixCompute(shaderModelMatrixCompute)
    , _batchSize(maxShaderStorageBlockSize / sizeof(glm::mat4)) {
    ZoneScoped;
    std::vector<Render_Queue::Command> commands;

    auto numCommands = rq->GetCommands().size();

    std::vector<glm::mat4> modelTranslateMatrices;
    std::vector<glm::mat4> modelRotateMatrices;
    std::vector<glm::mat4> modelScaleMatrices;

    modelTranslateMatrices.reserve(numCommands);
    modelRotateMatrices.reserve(numCommands);
    modelScaleMatrices.reserve(numCommands);

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

            auto modelMatrixIndex = modelTranslateMatrices.size();
            instances.modelMatrixIndices.emplace_back(GLuint(modelMatrixIndex));

            modelTranslateMatrices.emplace_back(
                glm::translate(cmd.transform.position));
            modelRotateMatrices.emplace_back(
                glm::mat4_cast(cmd.transform.rotation));
            modelScaleMatrices.emplace_back(glm::scale(cmd.transform.scale));
            instances.indirects.push_back(indirect);
        }
    }

    // Calculate the multiplied model matrices on the GPU
    glUseProgram(_shaderModelMatrixCompute->Program());
    auto numMatrices = (GLuint)modelTranslateMatrices.size();
    gl::VBO inputBuffers[MODEL_MATRIX_COMPUTE_BINDING_INPUT_MAX];
    // Create SSBO where the final matrices will be stored
    glGenBuffers(1, &_modelMatrixBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _modelMatrixBuffer);
    glBufferStorage(
        GL_SHADER_STORAGE_BUFFER, numMatrices * sizeof(glm::mat4), nullptr, 0);

    // Create SSBOs for the inputs
    std::vector<glm::mat4>
        *matrixVectors[MODEL_MATRIX_COMPUTE_BINDING_INPUT_MAX]
        = { &modelTranslateMatrices, &modelRotateMatrices,
            &modelScaleMatrices };
    void *inputBufferMapView[MODEL_MATRIX_COMPUTE_BINDING_INPUT_MAX]
        = { 0, 0, 0 };
    std::thread inputBufferCopyThreads[MODEL_MATRIX_COMPUTE_BINDING_INPUT_MAX];
    auto matrixVectorSize = numMatrices * sizeof(glm::mat4);

    for (int i = MODEL_MATRIX_COMPUTE_BINDING_START;
         i < MODEL_MATRIX_COMPUTE_BINDING_INPUT_MAX; i++) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, inputBuffers[i]);
        glBufferStorage(
            GL_SHADER_STORAGE_BUFFER, matrixVectorSize, nullptr,
            GL_MAP_WRITE_BIT);
        inputBufferMapView[i] = glMapBufferRange(
            GL_SHADER_STORAGE_BUFFER, 0, matrixVectorSize, GL_MAP_WRITE_BIT);

        // TODO(danielm): pool these threads
        inputBufferCopyThreads[i] = std::thread([&, i]() {
            memcpy(
                inputBufferMapView[i], matrixVectors[i]->data(),
                matrixVectorSize);
        });
    }

    gl::SetUniformLocation(
        _shaderModelMatrixCompute->TotalItemCount(), numMatrices);

    for (int i = MODEL_MATRIX_COMPUTE_BINDING_START;
         i < MODEL_MATRIX_COMPUTE_BINDING_INPUT_MAX; i++) {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, inputBuffers[i]);
    }
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _modelMatrixBuffer);

    auto const numItems = 16;
    auto numGroups = (numMatrices + numItems - 1) / numItems;

    // Wait until all of the input data is transfered
    for (int i = MODEL_MATRIX_COMPUTE_BINDING_START;
         i < MODEL_MATRIX_COMPUTE_BINDING_INPUT_MAX; i++) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, inputBuffers[i]);
        inputBufferCopyThreads[i].join();
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    }

    glDispatchComputeGroupSizeARB(numGroups, 1, 1, numItems, 1, 1);

    // Upload the draw indirect and matrix index buffers
    for (auto &materialGroup : _drawData) {
        for (auto &materialInstance : materialGroup.second) {
            for (auto &indexType : materialInstance.second.instances) {
                size_t remain = indexType.second.indirects.size();
                size_t next = 0;
                size_t currentBatchSize = 0;
                auto &indirects = indexType.second.indirects;
                auto &modelMatrixIndices = indexType.second.modelMatrixIndices;

                while (remain > 0) {
                    currentBatchSize
                        = (remain > _batchSize) ? _batchSize : remain;

                    GLuint bufIndirect, bufModelMatrixIndices;
                    glGenBuffers(1, &bufIndirect);
                    glGenBuffers(1, &bufModelMatrixIndices);

                    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, bufIndirect);
                    auto size = currentBatchSize * sizeof(GL_Indirect);
                    glBufferStorage(
                        GL_DRAW_INDIRECT_BUFFER, size, indirects.data() + next,
                        0);

                    glBindBuffer(
                        GL_SHADER_STORAGE_BUFFER, bufModelMatrixIndices);
                    glBufferStorage(
                        GL_SHADER_STORAGE_BUFFER,
                        currentBatchSize * sizeof(modelMatrixIndices[0]),
                        modelMatrixIndices.data() + next, 0);

                    indexType.second.indirectBuffers.emplace_back(bufIndirect);
                    indexType.second.modelMatrixIndexBuffers.emplace_back(
                        bufModelMatrixIndices);

                    remain -= currentBatchSize;
                    next += currentBatchSize;
                }

                indexType.second.lastBatchSize = currentBatchSize;
            }
        }
    }

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

GL_Multidraw::~GL_Multidraw() {
    ZoneScoped;
    glDeleteBuffers(1, &_modelMatrixBuffer);
    for (auto &materialGroup : _drawData) {
        for (auto &materialInstance : materialGroup.second) {
            for (auto &indexType : materialInstance.second.instances) {
                glDeleteBuffers(
                    indexType.second.indirectBuffers.size(),
                    indexType.second.indirectBuffers.data());
                glDeleteBuffers(
                    indexType.second.modelMatrixIndexBuffers.size(),
                    indexType.second.modelMatrixIndexBuffers.data());
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
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _modelMatrixBuffer);
        for (auto &materialInstance : materialGroup.second) {
            setupMaterial(materialGroup.first, materialInstance.first);
            for (auto &indexType : materialInstance.second.instances) {
                auto N = indexType.second.indirectBuffers.size();

                for (size_t i = 0; i < N; i++) {
                    auto &bufIndirect = indexType.second.indirectBuffers[i];
                    auto &modelMatrixIndexBuffer
                        = indexType.second.modelMatrixIndexBuffers[i];
                    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, bufIndirect);
                    glBindBufferBase(
                        GL_SHADER_STORAGE_BUFFER, 1, modelMatrixIndexBuffer);
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
    auto csh = FromStringLoadShader<GL_COMPUTE_SHADER>(
        model_matrix_compute_glsl, defines);

    auto builder = gl::Shader_Program_Builder();
    auto program = builder.Attach(csh).Link();
    if (program) {
        _program = std::move(program.value());

        _locTotalItemCount
            = gl::Uniform_Location<GLuint>(_program, "uiTotalItemCount");
    } else {
        throw Shader_Linker_Exception("textured lit", builder.Error());
    }
}

}
