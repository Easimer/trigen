// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "gl_depth_prepass.h"
#include "gl_shadercompiler.h"
#include "shader_program_builder.h"

#include <virtfs.hpp>

#include <unordered_map>

extern "C" {
    extern char const* depth_pass_vsh_glsl;
    extern char const* depth_pass_fsh_glsl;
}

VIRTFS_REGISTER_RESOURCE("shaders/depth_pass.vsh.glsl", depth_pass_vsh_glsl);
VIRTFS_REGISTER_RESOURCE("shaders/depth_pass.fsh.glsl", depth_pass_fsh_glsl);

namespace topo {

struct GL_Indirect {
    GLuint count;
    GLuint instanceCount;
    GLuint firstIndex;
    GLuint baseVertex;
    GLuint baseInstance;
};

GL_Depth_Prepass::GL_Depth_Prepass(
    GL_Model_Manager *modelManager,
    Renderable_Manager *renderableManager,
    unsigned width,
    unsigned height,
    GL_Depth_Pass_Shader *shader)
    : _modelManager(modelManager)
    , _renderableManager(renderableManager)
    , _shader(shader)
    , _width(width)
    , _height(height) {
    glBindTexture(GL_TEXTURE_2D, _depthMap);
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0,
        GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glBindFramebuffer(GL_FRAMEBUFFER, _framebuffer);
    glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, _depthMap, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void
GL_Depth_Prepass::Execute(Render_Queue *renderQueue, glm::mat4 matVP) {
    glBindFramebuffer(GL_FRAMEBUFFER, _framebuffer);
    glClear(GL_DEPTH_BUFFER_BIT);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);
    glUseProgram(_shader->Program());

    gl::SetUniformLocation(_shader->locVP(), matVP);

    _modelManager->BindMegabuffer();

    auto &commands = renderQueue->GetCommands();
    auto cmdIt = commands.cbegin();
    auto cmdEnd = commands.cend();

    std::unordered_map<GLenum, std::vector<glm::mat4>> modelMatrices;
    std::unordered_map<GLenum, std::vector<GL_Indirect>> elementTypeIndirectMap;

    for (auto &cmd : renderQueue->GetCommands()) {
        auto &transform = cmd.transform;
        auto renderable = cmd.renderable;
        if (_renderableManager->GetRenderableKind(renderable)
            != Renderable_Manager::RENDERABLE_MODEL) {
            continue;
        }

        Model_ID model;
        Material_ID material;
        _renderableManager->GetModelAndMaterial(renderable, &model, &material);

        GLuint indexOffset;
        GLuint firstIndex;
        GLuint baseVertex;
        GLenum elementType;
        size_t numElements;

        _modelManager->GetDrawParameters(
            model, &indexOffset, &baseVertex, &elementType, &numElements,
            &firstIndex);

        if (elementTypeIndirectMap.count(elementType) == 0) {
            elementTypeIndirectMap[elementType] = {};
            modelMatrices[elementType] = {};
        }

        GL_Indirect indirect;
        indirect.baseInstance = 0;
        indirect.baseVertex = baseVertex;
        indirect.count = numElements;
        indirect.instanceCount = 1;
        indirect.firstIndex = firstIndex;
        elementTypeIndirectMap[elementType].push_back(indirect);

        auto matTransform = glm::translate(cmd.transform.position)
            * glm::mat4_cast(cmd.transform.rotation)
            * glm::scale(cmd.transform.scale);
        modelMatrices[elementType].push_back(matTransform);
    }

    std::vector<GLuint> buffers;

    for (auto &eti : elementTypeIndirectMap) {
        auto &modelMatricesVec = modelMatrices[eti.first];
        auto &indirects = eti.second;
        size_t cmdNext = 0;
        size_t cmdRemain = eti.second.size();

        const size_t BATCH_SIZE = 256;

        while (cmdRemain > 0) {
            auto currentBatchSize = (cmdRemain > BATCH_SIZE) ? BATCH_SIZE : cmdRemain;

            GLuint bufIndirect, bufUniformMatrices;

            glGenBuffers(1, &bufIndirect);
            glBindBuffer(GL_DRAW_INDIRECT_BUFFER, bufIndirect);

            glGenBuffers(1, &bufUniformMatrices);
            glBindBuffer(GL_UNIFORM_BUFFER, bufUniformMatrices);
            glBufferData(
                GL_UNIFORM_BUFFER,
                currentBatchSize * sizeof(modelMatricesVec[0]),
                modelMatricesVec.data() + cmdNext, GL_STREAM_DRAW);
            glBindBufferBase(GL_UNIFORM_BUFFER, 0, bufUniformMatrices);
            glBindBuffer(GL_UNIFORM_BUFFER, 0);

            auto flags = GL_MAP_WRITE_BIT;

            auto size = currentBatchSize * sizeof(GL_Indirect);
            glBufferStorage(
                GL_DRAW_INDIRECT_BUFFER, size,
                nullptr, flags);
            auto indirectsDevice = glMapBufferRange(GL_DRAW_INDIRECT_BUFFER, 0, size, flags);
            auto indirectsHost = indirects.data() + cmdNext;
            memcpy(indirectsDevice, indirectsHost, size);
            glUnmapBuffer(GL_DRAW_INDIRECT_BUFFER);

            glMultiDrawElementsIndirect(
                GL_TRIANGLES, eti.first, nullptr, currentBatchSize, 0);

            cmdNext += currentBatchSize;
            cmdRemain -= currentBatchSize;

            buffers.push_back(bufIndirect);
            buffers.push_back(bufUniformMatrices);
        }
    }

    glDeleteBuffers(buffers.size(), buffers.data());

    return;

    // Draw lines

    GLuint vaoLines;
    glGenVertexArrays(1, &vaoLines);
    glBindVertexArray(vaoLines);

    for (auto &cmd : renderQueue->GetCommands()) {
        if (_renderableManager->GetRenderableKind(cmd.renderable)
            != Renderable_Manager::RENDERABLE_LINES) {
            continue;
        }

        glm::vec3 const *endpoints;
        glm::vec3 color;
        size_t lineCount;
        _renderableManager->GetLines(
            cmd.renderable, &endpoints, &lineCount, &color);
        auto size = lineCount * 2 * sizeof(glm::vec3);

        GLuint bufPosition = 0;

        glGenBuffers(1, &bufPosition);
        glBindBuffer(GL_ARRAY_BUFFER, bufPosition);
        glBufferData(GL_ARRAY_BUFFER, size, endpoints, GL_STREAM_DRAW);

        auto matTransform = glm::translate(cmd.transform.position)
            * glm::mat4_cast(cmd.transform.rotation)
            * glm::scale(cmd.transform.scale);
        gl::SetUniformLocation(_shader->locVP(), matVP);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);

        glDrawArrays(GL_LINES, 0, 2 * lineCount);

        glDeleteBuffers(1, &bufPosition);
    }

    glDeleteVertexArrays(1, &vaoLines);
}

void
GL_Depth_Prepass::BlitDepth(GLuint destFramebuffer) {
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, destFramebuffer);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, _framebuffer);

    glBlitFramebuffer(
        0, 0, _width, _height, 0, 0, _width, _height, GL_DEPTH_BUFFER_BIT,
        GL_NEAREST);
}

void
GL_Depth_Pass_Shader::build() {
    printf("[ topo ] building depth pass shader \n");
    char const *sourceVsh, *sourceFsh;

    virtfs::GetFile("shaders/depth_pass.vsh.glsl", &sourceVsh);
    virtfs::GetFile("shaders/depth_pass.fsh.glsl", &sourceFsh);

    auto vsh
        = FromStringLoadShader<GL_VERTEX_SHADER>(sourceVsh);
    auto fsh
        = FromStringLoadShader<GL_FRAGMENT_SHADER>(sourceFsh);

    auto builder = gl::Shader_Program_Builder("depth pass");
    auto program = builder.Attach(vsh).Attach(fsh).Link();
    if (program) {
        _program = std::move(program.value());

        _locVP = gl::Uniform_Location<glm::mat4>(_program, "matVP");
    } else {
        throw Shader_Linker_Exception("depth pass", builder.Error());
    }
}

}
