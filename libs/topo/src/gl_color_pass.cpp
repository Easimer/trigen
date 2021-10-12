// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "gl_color_pass.h"

#include "glres.h"
#include "shader_program_builder.h"

#include <Tracy.hpp>

extern "C" {
extern char const *solid_color_vsh_glsl;
extern char const *solid_color_fsh_glsl;
extern char const *lines2_vsh_glsl;
extern char const *lines2_fsh_glsl;

extern char const *textured_unlit_vsh_glsl;
extern char const *textured_unlit_fsh_glsl;
extern char const *textured_lit_vsh_glsl;
extern char const *textured_lit_fsh_glsl;
}

namespace topo {

struct GL_Indirect {
    GLuint count;
    GLuint instanceCount;
    GLuint firstIndex;
    GLuint baseVertex;
    GLuint baseInstance;
};

GL_Color_Pass::GL_Color_Pass(
    GL_Model_Manager *modelManager,
    Renderable_Manager *renderableManager,
    Material_Manager *materialManager,
    GL_Texture_Manager *textureManager,
    Shader_Textured_Unlit *shaderTexturedUnlit,
    Shader_Textured_Lit *shaderTexturedLit,
    Shader_Solid_Color *shaderSolidColor,
    Shader_Lines *shaderLines)
    : _modelManager(modelManager)
    , _renderableManager(renderableManager)
    , _shaderTexturedUnlit(shaderTexturedUnlit)
    , _shaderTexturedLit(shaderTexturedLit)
    , _shaderSolidColor(shaderSolidColor)
    , _shaderLines(shaderLines)
    , _materialManager(materialManager)
    , _textureManager(textureManager) { }

void
GL_Color_Pass::Execute(Render_Queue *renderQueue, GL_Multidraw &multiDraw, glm::mat4 const &matVP) {
    ZoneScoped;

    if (glPushDebugGroup) {
        glPushDebugGroup(GL_DEBUG_SOURCE_THIRD_PARTY, 1, -1, "Color pass");
    }

    std::vector<Render_Queue::Command> linesCommands;

    for (auto &cmd : renderQueue->GetCommands()) {
        switch (_renderableManager->GetRenderableKind(cmd.renderable)) {
        case Renderable_Manager::RENDERABLE_LINES:
            linesCommands.push_back(cmd);
            break;
        }
    }

    RenderLines(linesCommands, matVP);

    RenderModels(multiDraw, matVP);

    if (glPushDebugGroup) {
        glPopDebugGroup();
    }
}

void
GL_Color_Pass::RenderModels(
    GL_Multidraw &multiDraw,
    glm::mat4 const &matVP) {
    ZoneScoped;
    _modelManager->BindMegabuffer();

    auto setupShader = [&](topo::Material_Type type) {
        ZoneScoped;
        switch (type) {
        case topo::MAT_UNLIT_TRANSPARENT:
            // TODO:
            glUseProgram(_shaderTexturedUnlit->Program());
            gl::SetUniformLocation(_shaderTexturedUnlit->locMatVP(), matVP);
            break;
        case topo::MAT_UNLIT:
            glUseProgram(_shaderTexturedUnlit->Program());
            gl::SetUniformLocation(_shaderTexturedUnlit->locMatVP(), matVP);
            break;
        case topo::MAT_LIT:
            glUseProgram(_shaderTexturedLit->Program());
            gl::SetUniformLocation(_shaderTexturedLit->locMatVP(), matVP);
            break;
        case topo::MAT_SOLID_COLOR:
            glUseProgram(_shaderSolidColor->Program());
            gl::SetUniformLocation(_shaderSolidColor->locMatVP(), matVP);
            break;
        }
    };

    auto setupMaterial = [&](topo::Material_Type type,
                             topo::Material_ID material) {
        ZoneScoped;
        switch (type) {
        case topo::MAT_UNLIT: {
            auto *mul
                = (Material_Unlit *)_materialManager->GetMaterialData(material);
            auto texDiffuse = _textureManager->GetHandle(mul->diffuse);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texDiffuse);
            gl::SetUniformLocation(_shaderTexturedUnlit->locTexDiffuse(), 0);
            break;
        }
        case topo::MAT_UNLIT_TRANSPARENT: {
            // TODO:
            auto *mul
                = (Material_Transparent *)_materialManager->GetMaterialData(material);
            auto texDiffuse = _textureManager->GetHandle(mul->diffuse);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texDiffuse);
            gl::SetUniformLocation(_shaderTexturedUnlit->locTexDiffuse(), 0);
            break;
        }
        case topo::MAT_SOLID_COLOR: {
            auto *msc
                = (Material_Solid_Color *)_materialManager->GetMaterialData(
                    material);
            gl::SetUniformLocation(
                _shaderSolidColor->locColor(),
                { msc->color[0], msc->color[1], msc->color[2] });
            break;
        }
        case topo::MAT_LIT: {
            auto *ml
                = (Material_Lit *)_materialManager->GetMaterialData(
                    material);
            auto texDiffuse = _textureManager->GetHandle(ml->diffuse);
            auto texNormal = _textureManager->GetHandle(ml->normal);
            glActiveTexture(GL_TEXTURE0 + 0);
            glBindTexture(GL_TEXTURE_2D, texDiffuse);
            glActiveTexture(GL_TEXTURE0 + 1);
            glBindTexture(GL_TEXTURE_2D, texNormal);
            gl::SetUniformLocation(_shaderTexturedLit->locTexDiffuse(), 0);
            gl::SetUniformLocation(_shaderTexturedLit->locTexNormal(), 1);
            break;
        }
        }
    };

    multiDraw.Execute(setupShader, setupMaterial);
}

void
GL_Color_Pass::RenderLines(
    std::vector<Render_Queue::Command> const &commands,
    glm::mat4 const &matVP) {
    ZoneScoped;
    glUseProgram(_shaderLines->Program());

    GLuint vaoLines;
    glGenVertexArrays(1, &vaoLines);
    glBindVertexArray(vaoLines);

    for (auto &cmd : commands) {
        assert(
            _renderableManager->GetRenderableKind(cmd.renderable)
            == Renderable_Manager::RENDERABLE_LINES);

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
        gl::SetUniformLocation(_shaderLines->locMatMVP(), matVP * matTransform);
        gl::SetUniformLocation(_shaderLines->locColor(), color);

        glVertexAttribPointer(
            0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
        glEnableVertexAttribArray(0);

        glDrawArrays(GL_LINES, 0, 2 * lineCount);

        glDeleteBuffers(1, &bufPosition);
    }

    glDeleteVertexArrays(1, &vaoLines);
}

void
Shader_Solid_Color::Build() {
    printf("[ topo ] building shader 'solid color'\n");
    auto vsh = FromStringLoadShader<GL_VERTEX_SHADER>(solid_color_vsh_glsl);
    auto fsh = FromStringLoadShader<GL_FRAGMENT_SHADER>(solid_color_fsh_glsl);

    auto builder = gl::Shader_Program_Builder();
    auto program = builder.Attach(vsh).Attach(fsh).Link();
    if (program) {
        _program = std::move(program.value());

        _locMatVP = gl::Uniform_Location<glm::mat4>(_program, "matVP");
        _locColor = gl::Uniform_Location<glm::vec3>(_program, "solidColor");
    } else {
        throw Shader_Linker_Exception("solid color", builder.Error());
    }
}

void
Shader_Lines::Build() {
    printf("[ topo ] building shader 'lines'\n");
    auto vsh = FromStringLoadShader<GL_VERTEX_SHADER>(lines2_vsh_glsl);
    auto fsh = FromStringLoadShader<GL_FRAGMENT_SHADER>(lines2_fsh_glsl);

    auto builder = gl::Shader_Program_Builder();
    auto program = builder.Attach(vsh).Attach(fsh).Link();
    if (program) {
        _program = std::move(program.value());

        _locMatMVP = gl::Uniform_Location<glm::mat4>(_program, "matMVP");
        _locColor = gl::Uniform_Location<glm::vec3>(_program, "color");
    } else {
        throw Shader_Linker_Exception("lines", builder.Error());
    }
}

void
Shader_Textured_Unlit::Build() {
    printf("[ topo ] building shader 'textured unlit'\n");
    auto vsh = FromStringLoadShader<GL_VERTEX_SHADER>(textured_unlit_vsh_glsl);
    auto fsh = FromStringLoadShader<GL_FRAGMENT_SHADER>(textured_unlit_fsh_glsl);

    auto builder = gl::Shader_Program_Builder();
    auto program = builder.Attach(vsh).Attach(fsh).Link();
    if (program) {
        _program = std::move(program.value());

        _locMatVP = gl::Uniform_Location<glm::mat4>(_program, "matVP");
        _locTexDiffuse = gl::Uniform_Location<GLint>(_program, "texDiffuse");
    } else {
        throw Shader_Linker_Exception("textured unlit", builder.Error());
    }
}

void
Shader_Textured_Lit::Build() {
    printf("[ topo ] building shader 'textured lit'\n");
    auto vsh = FromStringLoadShader<GL_VERTEX_SHADER>(textured_lit_vsh_glsl);
    auto fsh = FromStringLoadShader<GL_FRAGMENT_SHADER>(textured_lit_fsh_glsl);

    auto builder = gl::Shader_Program_Builder();
    auto program = builder.Attach(vsh).Attach(fsh).Link();
    if (program) {
        _program = std::move(program.value());

        _locMatVP = gl::Uniform_Location<glm::mat4>(_program, "matVP");
        _locTexDiffuse = gl::Uniform_Location<GLint>(_program, "texDiffuse");
        _locTexNormal = gl::Uniform_Location<GLint>(_program, "texNormal");
    } else {
        throw Shader_Linker_Exception("textured lit", builder.Error());
    }
}

}
