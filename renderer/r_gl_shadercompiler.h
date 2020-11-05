// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: OpenGL shader compilation helpers
//

#pragma once

#include <optional>
#include <vector>
#include <string>
#include <trigen/glres.h>

struct Shader_Define {
    std::string key;
    std::string value;
};

using Shader_Define_List = std::vector<Shader_Define>;

template<typename Shader>
bool CompileShaderFromString(Shader const& shader, char const* pszSource, char const* pszPath, Shader_Define_List const& defines) {
    GLint bSuccess;
    std::vector<std::string> defines_fmt;
    std::vector<char const*> sources;

    bool is_mesa_gpu = false;

    // Detect open-source Intel drivers
    char const* vendor = (char*)glGetString(GL_VENDOR);
    is_mesa_gpu |= (strcmp(vendor, "Intel Open Source Technology Center") == 0);
    is_mesa_gpu |= (strcmp(vendor, "VMware, Inc.") == 0);

    char const* pszVersion = "#version 330 core\n";
    char const* pszLineReset = "#line -1\n";

    if (is_mesa_gpu) {
        pszVersion = "#version 130\n";
    }

    sources.push_back(pszVersion);

    if (is_mesa_gpu) {
        sources.push_back("#define VAO_LAYOUT(i)\n");
    }

    for (auto& def : defines) {
        char buf[64];
        snprintf(buf, 63, "#define %s %s\n", def.key.c_str(), def.value.c_str());
        defines_fmt.push_back(std::string((char const*)buf));
        sources.push_back(defines_fmt.back().c_str());
    }

    if (!is_mesa_gpu) {
        sources.push_back(pszLineReset);
    }

    sources.push_back(pszSource);

    glShaderSource(shader, sources.size(), sources.data(), NULL);
    glCompileShader(shader);
    glGetShaderiv(shader, GL_COMPILE_STATUS, &bSuccess);

    if (bSuccess == 0) {
        char pchMsgBuf[256];
        glGetShaderInfoLog(shader, 256, NULL, pchMsgBuf);
        if (defines.size() > 0) {
            printf("Compilation of shader '%s', with defines:\n", pszPath);
            for (auto& def : defines) {
                printf("\t%s = %s\n", def.key.c_str(), def.value.c_str());
            }
            printf("has failed:\n%s\n", pchMsgBuf);
        }
        else {
            printf("Compilation of shader '%s' has failed:\n%s\n", pszPath, pchMsgBuf);
        }
    }

    return bSuccess != 0;
}

template<GLenum kType>
std::optional<gl::Shader<kType>> FromStringLoadShader(char const* pszSource, Shader_Define_List const& defines) {
    gl::Shader<kType> shader;

    if (!CompileShaderFromString(shader, pszSource, "<string>", defines)) {
        return std::nullopt;
    }

    return shader;
}

template<GLenum kType>
std::optional<gl::Shader<kType>> FromStringLoadShader(char const* pszSource) {
    Shader_Define_List x;

    return FromStringLoadShader<kType>(pszSource, x);
}