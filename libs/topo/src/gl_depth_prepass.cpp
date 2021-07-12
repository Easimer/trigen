// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "gl_depth_prepass.h"
#include "gl_shadercompiler.h"
#include "shader_program_builder.h"

#include <virtfs.hpp>

extern "C" {
    extern char const* depth_pass_vsh_glsl;
    extern char const* depth_pass_fsh_glsl;
}

VIRTFS_REGISTER_RESOURCE("shaders/depth_pass.vsh.glsl", depth_pass_vsh_glsl);
VIRTFS_REGISTER_RESOURCE("shaders/depth_pass.fsh.glsl", depth_pass_fsh_glsl);

namespace topo {

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
    glUseProgram(_shader->Program());

    _modelManager->BindMegabuffer();

    std::vector<Renderable_ID> models;

    for (auto &cmd : renderQueue->GetCommands()) {
        if (_renderableManager->GetRenderableKind(cmd.renderable)
            != Renderable_Manager::RENDERABLE_MODEL) {
            continue;
        }

        Model_ID model;
        Material_ID material;
        _renderableManager->GetModelAndMaterial(
            cmd.renderable, &model, &material);

        auto matTransform = glm::translate(cmd.transform.position)
            * glm::mat4_cast(cmd.transform.rotation)
            * glm::scale(cmd.transform.scale);
        gl::SetUniformLocation(_shader->locMVP(), matVP * matTransform);

        void *indexOffset;
        GLint baseVertex;
        GLenum elementType;
        size_t numElements;

        _modelManager->GetDrawParameters(
            model, &indexOffset, &baseVertex, &elementType, &numElements);

        glDrawElementsBaseVertex(
            GL_TRIANGLES, numElements, elementType, indexOffset, baseVertex);
    }

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
        gl::SetUniformLocation(_shader->locMVP(), matVP * matTransform);

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

        _locMVP = gl::Uniform_Location<glm::mat4>(_program, "matMVP");
    } else {
        throw Shader_Linker_Exception("depth pass", builder.Error());
    }
}

}
