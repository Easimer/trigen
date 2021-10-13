// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "gl_depth_prepass.h"
#include "gl_shadercompiler.h"
#include "shader_program_builder.h"

#include <virtfs.hpp>

#include <unordered_map>

#include <Tracy.hpp>

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

GL_Depth_Pass::GL_Depth_Pass(
    std::string const &name,
    GL_Model_Manager *modelManager,
    Renderable_Manager *renderableManager,
    unsigned width,
    unsigned height,
    GL_Depth_Pass_Shader *shader)
    : _name(name)
    , _modelManager(modelManager)
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
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    float borderColor[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);

    glBindFramebuffer(GL_FRAMEBUFFER, _framebuffer);
    glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, _depthMap, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void
GL_Depth_Pass::Clear() {
    ZoneScoped;
    glBindFramebuffer(GL_FRAMEBUFFER, _framebuffer);
    glClear(GL_DEPTH_BUFFER_BIT);
}

void
GL_Depth_Pass::Execute(Render_Queue *renderQueue, GL_Multidraw &multiDraw, glm::mat4 matVP) {
    ZoneScoped;

    if (glPushDebugGroup) {
        glPushDebugGroup(GL_DEBUG_SOURCE_THIRD_PARTY, 1, _name.size(), _name.c_str());
    }

    glBindFramebuffer(GL_FRAMEBUFFER, _framebuffer);
    glClear(GL_DEPTH_BUFFER_BIT);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);
    glUseProgram(_shader->Program());

    gl::SetUniformLocation(_shader->locVP(), matVP);

    _modelManager->BindMegabuffer();

    auto setupShader = [&](topo::Material_Type type) {
        switch (type) {
        case topo::Material_Type::MAT_UNLIT_TRANSPARENT:
            return false;
        default:
            return true;
        }
    };

    auto setupMaterial
        = [&](topo::Material_Type, topo::Material_ID) { return true; };

    multiDraw.Execute(setupShader, setupMaterial);

    // Draw lines

    GLuint vaoLines;
    glGenVertexArrays(1, &vaoLines);
    glBindVertexArray(vaoLines);

    for (auto &cmd : renderQueue->GetCommands()) {
        if (_renderableManager->GetRenderableKind(cmd.renderable)
            != Renderable_Manager::RENDERABLE_LINES) {
            continue;
        }

        GLuint zero(0);
        glm::vec3 const *endpoints;
        glm::vec3 color;
        size_t lineCount;
        _renderableManager->GetLines(
            cmd.renderable, &endpoints, &lineCount, &color);
        auto size = lineCount * 2 * sizeof(glm::vec3);

        gl::VBO bufPosition, bufUniformMatrices, bufModelMatrixIndices;

        glBindBuffer(GL_ARRAY_BUFFER, bufPosition);
        glBufferData(GL_ARRAY_BUFFER, size, endpoints, GL_STREAM_DRAW);

        auto matTransform = glm::translate(cmd.transform.position)
            * glm::mat4_cast(cmd.transform.rotation)
            * glm::scale(cmd.transform.scale);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufUniformMatrices);
        glBufferStorage(
            GL_SHADER_STORAGE_BUFFER, sizeof(glm::mat4),
            glm::value_ptr(matTransform), 0);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufUniformMatrices);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufModelMatrixIndices);
        glBufferStorage(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint), &zero, 0);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bufModelMatrixIndices);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);

        glDrawArrays(GL_LINES, 0, 2 * lineCount);
    }

    glDeleteVertexArrays(1, &vaoLines);

    if (glPushDebugGroup) {
        glPopDebugGroup();
    }
}

void
GL_Depth_Pass::BlitDepth(GLuint destFramebuffer) {
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
