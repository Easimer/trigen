// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: generic shader
//

#include "shader_generic.h"
#include "gl_shadercompiler.h"
#include "shader_program_builder.h"

extern "C" {
extern char const *generic_vsh_glsl;
extern char const *generic_fsh_glsl;
}

namespace topo {
// ========================== Shader_Generic_Base

void
Shader_Generic_Base::build() {
    printf("renderer: building shader '%s'\n", name());
    auto vsh
        = FromStringLoadShader<GL_VERTEX_SHADER>(generic_vsh_glsl, defines());
    auto fsh
        = FromStringLoadShader<GL_FRAGMENT_SHADER>(generic_fsh_glsl, defines());

    auto builder = gl::Shader_Program_Builder(name());
    auto program = builder.Attach(vsh).Attach(fsh).Link();
    if (program) {
        _program = std::move(program.value());

        linkUniforms();
    } else {
        throw Shader_Linker_Exception(name(), builder.Error());
    }
}

void
Shader_Generic_Base::linkUniforms() {
    _locMVP = gl::Uniform_Location<glm::mat4>(program(), "matMVP");
    _locTintColor = gl::Uniform_Location<glm::vec4>(program(), "tintColor");
}

// ========================== Shader_Generic

char const *
Shader_Generic::name() const {
    return "Generic";
}

GL_Shader_Define_List
Shader_Generic::defines() const {
    return {};
}

// ========================== Shader_Generic_With_Vertex_Colors

char const *
Shader_Generic_With_Vertex_Colors::name() const {
    return "Generic (with vertex colors)";
}

GL_Shader_Define_List
Shader_Generic_With_Vertex_Colors::defines() const {
    return { { "GENERIC_SHADER_WITH_VERTEX_COLORS", "1" } };
}

// ========================== Shader_Generic_Textured_Unlit

char const *
Shader_Generic_Textured_Unlit::name() const {
    return "Generic (textured, unlit)";
}

GL_Shader_Define_List
Shader_Generic_Textured_Unlit::defines() const {
    return { { "TEXTURED", "1" } };
}

void
Shader_Generic_Textured_Unlit::linkUniforms() {
    Shader_Generic::linkUniforms();

    _locTexDiffuse = gl::Uniform_Location<GLint>(program(), "texDiffuse");
}

// ========================== Shader_Generic_Textured_Lit

char const *
Shader_Generic_Textured_Lit::name() const {
    return "Generic (textured, lit)";
}

GL_Shader_Define_List
Shader_Generic_Textured_Lit::defines() const {
    return { { "TEXTURED", "1" }, { "LIT", "1" } };
}

void
Shader_Generic_Textured_Lit::linkUniforms() {
    Shader_Generic::linkUniforms();

    _locTexNormal = gl::Uniform_Location<GLint>(program(), "texNormal");
    _locSunPosition = gl::Uniform_Location<glm::vec3>(program(), "sunPosition");
    _locModelMatrix = gl::Uniform_Location<glm::mat4>(program(), "matModel");
    _locViewPosition
        = gl::Uniform_Location<glm::vec3>(program(), "viewPosition");
}
}
