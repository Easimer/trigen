// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"

#include <glad/glad.h>

#include "gbuffer.h"
#include "r_gl_shadercompiler.h"
#include "shader_program_builder.h"

extern "C" {
    extern char const* deferred_vsh_glsl;
    extern char const* deferred_fsh_glsl;
}

G_Buffer::G_Buffer()
    : _prevFBDraw(0)
    , _prevFBRead(0) {
}

G_Buffer::G_Buffer(unsigned width, unsigned height) {
    if (glObjectLabel) {
        glObjectLabel(GL_FRAMEBUFFER, _fb, -1, "G-buffer");
        glObjectLabel(GL_TEXTURE, _bufBaseColor, -1, "G-buffer::base color");
        glObjectLabel(GL_TEXTURE, _bufNormal, -1, "G-buffer::normal");
        glObjectLabel(GL_TEXTURE, _bufPosition, -1, "G-buffer::position");
    }

    // Store the handles to the original framebuffers
    // The default framebuffer (id=0) may not be the framebuffer we're
    // supposed to draw into (e.g. the viewport is embedded into a Qt window)
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &_prevFBDraw);
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &_prevFBRead);

    glBindFramebuffer(GL_FRAMEBUFFER, _fb);

    glBindTexture(GL_TEXTURE_2D, _bufBaseColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _bufBaseColor, 0);

    glBindTexture(GL_TEXTURE_2D, _bufNormal);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, _bufNormal, 0);

    glBindTexture(GL_TEXTURE_2D, _bufPosition);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, _bufPosition, 0);

    glBindRenderbuffer(GL_RENDERBUFFER, _bufRender);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, _bufRender);

    GLuint attachments[3] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
    glDrawBuffers(3, attachments);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        printf("[ gfx ] G-buffer framebuffer wasn't complete!");
    }

    float const vertices[] = {
        -1.0f, -1.0f,
        +1.0f, -1.0f,
        -1.0f, +1.0f,
        +1.0f, +1.0f,
    };

    float const texcoords[] = {
        0, 0,
        1, 0,
        0, 1,
        1, 1,
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

        auto vsh = FromStringLoadShader<GL_VERTEX_SHADER>(deferred_vsh_glsl, {});
        auto fsh = FromStringLoadShader<GL_FRAGMENT_SHADER>(deferred_fsh_glsl, {});

        gl::Shader_Program_Builder builder("G-buffer shader");
        auto program = builder.Attach(vsh).Attach(fsh).Link();
        if (program) {
            auto locTexBaseColor = gl::Uniform_Location<GLint>(*program, "texBaseColor");
            auto locTexNormal = gl::Uniform_Location<GLint>(*program, "texNormal");
            auto locTexPosition = gl::Uniform_Location<GLint>(*program, "texPosition");
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

void G_Buffer::activate() {
    glBindFramebuffer(GL_FRAMEBUFFER, _fb);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void G_Buffer::draw() {
    if (!_program) {
        return;
    }

    glBindFramebuffer(GL_READ_FRAMEBUFFER, _prevFBRead);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, _prevFBDraw);

    glBindVertexArray(_quadVao);

    glUseProgram(_program->program);

    glActiveTexture(GL_TEXTURE0 + 0);
    glBindTexture(GL_TEXTURE_2D, _bufBaseColor);
    glActiveTexture(GL_TEXTURE0 + 1);
    glBindTexture(GL_TEXTURE_2D, _bufNormal);
    glActiveTexture(GL_TEXTURE0 + 2);
    glBindTexture(GL_TEXTURE_2D, _bufPosition);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}