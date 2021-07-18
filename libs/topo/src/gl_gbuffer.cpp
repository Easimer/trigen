// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"

#include <glad/glad.h>
#include <virtfs.hpp>

#include "gl_gbuffer.h"
#include "gl_shadercompiler.h"
#include "shader_program_builder.h"

#include <imgui.h>

extern "C" {
    extern char const* gbuffer_merge_vsh_glsl;
    extern char const* gbuffer_merge_fsh_glsl;
}

VIRTFS_REGISTER_RESOURCE("shaders/gbuffer_merge.vsh.glsl", gbuffer_merge_vsh_glsl);
VIRTFS_REGISTER_RESOURCE("shaders/gbuffer_merge.fsh.glsl", gbuffer_merge_fsh_glsl);

namespace topo {
struct Light {
    glm::vec4 position;
    glm::vec4 color;
};
struct Lights_Uniform_Block {
    glm::vec4 viewPosition;
    GLint numLights;
    char padding[12];

    Light lights[0];
};

static void
formatLabelAndApply(
    char const *name,
    char const *suffix,
    GLenum type,
    GLint handle) {
    char *labelBuf = nullptr;
    int res = snprintf(labelBuf, 0, "%s::%s", name, suffix);
    assert(res >= 0);
    labelBuf = new char[size_t(res) + 1];
    assert(labelBuf);
    snprintf(labelBuf, size_t(res) + 1, "%s::%s", name, suffix);
    glObjectLabel(type, handle, -1, labelBuf);
    delete[] labelBuf;
}

G_Buffer::G_Buffer(char const *name, unsigned width, unsigned height)
    : _width(width)
    , _height(height) {

    GLint prevFbDraw, prevFbRead;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prevFbDraw);
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &prevFbRead);

    glBindFramebuffer(GL_FRAMEBUFFER, _fb);

    glBindTexture(GL_TEXTURE_2D, _bufBaseColor);
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT,
        NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _bufBaseColor, 0);

    glBindTexture(GL_TEXTURE_2D, _bufNormal);
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT,
        NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, _bufNormal, 0);

    glBindTexture(GL_TEXTURE_2D, _bufPosition);
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT,
        NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, _bufPosition, 0);

    glBindRenderbuffer(GL_RENDERBUFFER, _bufRender);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
    glFramebufferRenderbuffer(
        GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER,
        _bufRender);

    GLuint attachments[3]
        = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
    glDrawBuffers(3, attachments);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        printf("[ topo ] G-buffer framebuffer wasn't complete!");
    }

    if (glObjectLabel) {
        formatLabelAndApply(name, "", GL_FRAMEBUFFER, _fb);
        formatLabelAndApply(name, "base color", GL_TEXTURE, _bufBaseColor);
        formatLabelAndApply(name, "normal", GL_TEXTURE, _bufNormal);
        formatLabelAndApply(name, "position", GL_TEXTURE, _bufPosition);
    }

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, prevFbDraw);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, prevFbRead);

    float const vertices[] = {
        -1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f, +1.0f, +1.0f,
    };

    float const texcoords[] = {
        0, 0, 1, 0, 0, 1, 1, 1,
    };

    glBindVertexArray(_quadVao);

    glBindBuffer(GL_ARRAY_BUFFER, _quadPosVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, _quadUvVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
    glEnableVertexAttribArray(1);

    try {
        GL_Shader_Define_List defines = { };
        char const *deferredVshGlsl = nullptr;
        char const *deferredFshGlsl = nullptr;
        virtfs::GetFile("shaders/gbuffer_merge.vsh.glsl", &deferredVshGlsl);
        virtfs::GetFile("shaders/gbuffer_merge.fsh.glsl", &deferredFshGlsl);
        auto vsh
            = FromStringLoadShader<GL_VERTEX_SHADER>(deferredVshGlsl, defines);
        auto fsh = FromStringLoadShader<GL_FRAGMENT_SHADER>(
            deferredFshGlsl, defines);

        gl::Shader_Program_Builder builder("G-buffer shader");
        auto program = builder.Attach(vsh).Attach(fsh).Link();
        if (program) {
            auto locTexBaseColor
                = gl::Uniform_Location<GLint>(*program, "texBaseColor");
            auto locTexNormal
                = gl::Uniform_Location<GLint>(*program, "texNormal");
            auto locTexPosition
                = gl::Uniform_Location<GLint>(*program, "texPosition");
            _program = {
                std::move(program.value()),
                locTexBaseColor,
                locTexNormal,
                locTexPosition,
            };
        }
    } catch (Shader_Compiler_Exception const &ex) {
        printf("%s\n", ex.errorMessage().c_str());
    }
}

void
G_Buffer::activate() {
    glBindFramebuffer(GL_FRAMEBUFFER, _fb);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void
G_Buffer::draw(
    Render_Queue *rq,
    glm::vec3 const &viewPosition,
    GLint readFramebuffer,
    GLint drawFramebuffer,
    unsigned screenWidth,
    unsigned screenHeight) {
    if (!_program) {
        return;
    }

    auto &lights = rq->GetLights();
    auto numLights = lights.size();

    auto lightsUBOSize = sizeof(Lights_Uniform_Block) + numLights * sizeof(Light);
    auto lightsUBOBufferU8 = std::make_unique<uint8_t[]>(lightsUBOSize);
    auto lightsUBOBuffer = (Lights_Uniform_Block *)lightsUBOBufferU8.get();

    lightsUBOBuffer->numLights = numLights;
    lightsUBOBuffer->viewPosition = glm::vec4(viewPosition, 1);

    for (size_t i = 0; i < numLights; i++) {
        lightsUBOBuffer->lights[i].color = lights[i].light;
        lightsUBOBuffer->lights[i].position = glm::vec4(lights[i].transform.position, 1);
    }

    GLuint bufLights;
    glGenBuffers(1, &bufLights);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufLights);
    glBufferData(
        GL_SHADER_STORAGE_BUFFER, lightsUBOSize, lightsUBOBuffer,
        GL_STREAM_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    glUseProgram(_program->program);

    glDepthMask(GL_FALSE);

    glBindVertexArray(_quadVao);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, readFramebuffer);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, drawFramebuffer);

    glActiveTexture(GL_TEXTURE0 + 0);
    glBindTexture(GL_TEXTURE_2D, _bufBaseColor);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glActiveTexture(GL_TEXTURE0 + 1);
    glBindTexture(GL_TEXTURE_2D, _bufNormal);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glActiveTexture(GL_TEXTURE0 + 2);
    glBindTexture(GL_TEXTURE_2D, _bufPosition);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    gl::SetUniformLocation(_program->texBaseColor, 0);
    gl::SetUniformLocation(_program->texNormal, 1);
    gl::SetUniformLocation(_program->texPosition, 2);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufLights);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, _fb);
    glBlitFramebuffer(
        0, 0, _width, _height, 0, 0, screenWidth, screenHeight,
        GL_DEPTH_BUFFER_BIT, GL_NEAREST);

    glDeleteBuffers(1, &bufLights);
    glDepthMask(GL_TRUE);
}
}