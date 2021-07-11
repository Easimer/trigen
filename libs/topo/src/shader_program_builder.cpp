// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: OpenGL shader program builder
//

#include "shader_program_builder.h"

namespace gl {

Shader_Program_Builder::Shader_Program_Builder()
    : _program()
    , _name("<unnamed>")
    , _errorMsg() {
}

Shader_Program_Builder::Shader_Program_Builder(std::string const &name)
    : _program()
    , _name(name)
    , _errorMsg() {
    if (glObjectLabel) {
        glObjectLabel(GL_PROGRAM, _program, -1, name.c_str());
    }
}

Shader_Program_Builder &Shader_Program_Builder::Attach(Vertex_Shader const &vsh) {
    glAttachShader(_program, vsh);
    return *this;
}

Shader_Program_Builder &Shader_Program_Builder::Attach(Fragment_Shader const &fsh) {
    glAttachShader(_program, fsh);
    return *this;
}

Optional<Shader_Program> Shader_Program_Builder::Link() {
    glLinkProgram(_program);
    GLint bSuccess;
    glGetProgramiv(_program, GL_LINK_STATUS, &bSuccess);
    if (bSuccess != 0) {
        return { std::move(_program) };
    } else {
        GLsizei msgLen = 0;
        glGetProgramiv(_program, GL_INFO_LOG_LENGTH, &msgLen);
        _errorMsg.resize(msgLen);
        glGetProgramInfoLog(_program, msgLen, NULL, _errorMsg.data());
    }

    return {};
}

char const *Shader_Program_Builder::Error() const {
    return _errorMsg.c_str();
}

char const *Shader_Program_Builder::Name() const {
    return _name.c_str();
}

}
