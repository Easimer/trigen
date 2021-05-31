// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: OpenGL shader compilation helpers
//

#pragma once

#include <optional>
#include <vector>
#include <string>
#include "glres.h"

struct Shader_Define {
    std::string key;
    std::string value;
};

using Shader_Define_List = std::vector<Shader_Define>;

class Shader_Compiler_Exception {
public:
    Shader_Compiler_Exception(std::string errorMessage, GLenum stageKind)
        : _errorMessage(std::move(errorMessage))
        , _stageKind(stageKind) {
    }

    std::string const &errorMessage() const { return _errorMessage; }
    GLenum stageKind() const { return _stageKind; }

private:
    std::string _errorMessage;
    GLenum _stageKind;
};

void CompileShaderFromString(GLuint shader, char const *pszSource, Shader_Define_List const &defines, GLenum stageKind);

template<typename Shader>
void CompileShaderFromString(Shader const& shader, char const* pszSource, Shader_Define_List const& defines, GLenum stageKind) {
    CompileShaderFromString((GLuint)shader, pszSource, defines, stageKind);
}

template<GLenum kType>
gl::Shader<kType> FromStringLoadShader(char const* pszSource, Shader_Define_List const& defines) {
    gl::Shader<kType> shader;

    CompileShaderFromString(shader, pszSource, defines, kType);

    return shader;
}

template<GLenum kType>
gl::Shader<kType> FromStringLoadShader(char const* pszSource) {
    Shader_Define_List x;

    return FromStringLoadShader<kType>(pszSource, x);
}
