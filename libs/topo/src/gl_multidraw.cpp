// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "gl_multidraw.h"

#include <Tracy.hpp>

namespace topo {

GL_Multidraw::GL_Multidraw(
    Render_Queue *rq,
    Renderable_Manager *renderableManager,
    Material_Manager *materialManager,
    GL_Model_Manager *modelManager) {
    ZoneScoped;
    std::vector<Render_Queue::Command> commands;

    GLint maxUniformBlockSize;
    glGetIntegerv(GL_MAX_UNIFORM_BLOCK_SIZE, &maxUniformBlockSize);

    _batchSize = maxUniformBlockSize / sizeof(glm::mat4);

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

            auto matModel = glm::translate(cmd.transform.position)
                * glm::mat4_cast(cmd.transform.rotation)
                * glm::scale(cmd.transform.scale);
            instances.modelMatrices.push_back(matModel);
            instances.indirects.push_back(indirect);
        }
    }

    // Upload the draw indirect and uniform buffers
    for (auto &materialGroup : _drawData) {
        for (auto& materialInstance : materialGroup.second) {
            for (auto &indexType : materialInstance.second.instances) {
                size_t remain = indexType.second.indirects.size();
                size_t next = 0;
                size_t currentBatchSize = 0;
                auto &indirects = indexType.second.indirects;
                auto &modelMatricesVec = indexType.second.modelMatrices;

                while (remain > 0) {
                    currentBatchSize
                        = (remain > _batchSize) ? _batchSize : remain;

                    GLuint bufIndirect, bufUniformMatrices;
                    glGenBuffers(1, &bufIndirect);
                    glGenBuffers(1, &bufUniformMatrices);

                    glBindBuffer(GL_UNIFORM_BUFFER, bufUniformMatrices);
                    glBufferData(
                        GL_UNIFORM_BUFFER,
                        currentBatchSize * sizeof(modelMatricesVec[0]),
                        modelMatricesVec.data() + next, GL_STREAM_DRAW);
                    glBindBuffer(GL_UNIFORM_BUFFER, 0);

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

                    indexType.second.modelMatrixBuffers.emplace_back(
                        std::move(bufUniformMatrices));
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
                    glBindBufferBase(GL_UNIFORM_BUFFER, 0, bufUniformMatrices);
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

}
