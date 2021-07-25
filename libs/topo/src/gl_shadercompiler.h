// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: OpenGL shader compilation helpers
//

#pragma once

#include <optional>
#include <vector>
#include <string>
#include "glres.h"
#include "gl_common.h"

namespace topo {
struct GL_Shader_Define {
    std::string key, value;
};

using GL_Shader_Define_List = std::vector<GL_Shader_Define>;

class Shader_Compiler_Exception : public std::exception {
public:
    Shader_Compiler_Exception(std::string errorMessage, GLenum stageKind)
        : _errorMessage(std::move(errorMessage))
        , _stageKind(stageKind) { }

    std::string const &
    errorMessage() const {
        return _errorMessage;
    }
    GLenum
    stageKind() const {
        return _stageKind;
    }

    char const *what() const noexcept {
        return _errorMessage.c_str();
    }

private:
    std::string _errorMessage;
    GLenum _stageKind;
};

class Shader_Linker_Exception {
public:
    Shader_Linker_Exception(std::string programName, std::string errorMessage)
        : _programName(std::move(programName))
        , _errorMessage(std::move(errorMessage)) { }

    std::string const &
    programName() const {
        return _programName;
    }
    std::string const &
    errorMessage() const {
        return _errorMessage;
    }

private:
    std::string _programName;
    std::string _errorMessage;
};

void
CompileShaderFromString(
    GLuint shader,
    char const *pszSource,
    GL_Shader_Define_List const &defines,
    GLenum stageKind);

template <typename Shader>
void
CompileShaderFromString(
    Shader const &shader,
    char const *pszSource,
    GL_Shader_Define_List const &defines,
    GLenum stageKind) {
    CompileShaderFromString((GLuint)shader, pszSource, defines, stageKind);
}

template <GLenum kType>
gl::Shader<kType>
FromStringLoadShader(
    char const *pszSource,
    GL_Shader_Define_List const &defines) {
    gl::Shader<kType> shader;

    CompileShaderFromString(shader, pszSource, defines, kType);

    return shader;
}

template <GLenum kType>
gl::Shader<kType>
FromStringLoadShader(char const *pszSource) {
    GL_Shader_Define_List x;

    return FromStringLoadShader<kType>(pszSource, x);
}
}
