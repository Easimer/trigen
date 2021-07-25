// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: OpenGL shader compilation helpers
//

#include "gl_shadercompiler.h"

namespace topo {

void
CompileShaderFromString(
    GLuint shader,
    char const *source,
    GL_Shader_Define_List const &defines,
    GLenum stageKind) {
    GLint bSuccess;
    std::vector<std::string> defines_fmt;
    std::vector<char const *> sources;

    char const *version = "#version 450 core\n";

    sources.push_back(version);

    for (auto &def : defines) {
        char buf[64];
        snprintf(
            buf, 63, "#define %s %s\n", def.key.c_str(), def.value.c_str());
        defines_fmt.push_back(std::string((char const *)buf));
        sources.push_back(defines_fmt.back().c_str());
    }

    sources.push_back(source);

    glShaderSource(shader, sources.size(), sources.data(), NULL);
    glCompileShader(shader);
    glGetShaderiv(shader, GL_COMPILE_STATUS, &bSuccess);

    if (bSuccess == 0) {
        std::string msgBuf;
        GLsizei msgLen = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &msgLen);
        msgBuf.resize(msgLen);
        glGetShaderInfoLog(shader, msgLen, NULL, msgBuf.data());

        throw Shader_Compiler_Exception(msgBuf, stageKind);
    }
}
}
