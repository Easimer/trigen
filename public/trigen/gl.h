// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: GL Library
//

#pragma once

namespace gl {
    namespace detail {
        template<typename T>
        void _Bind(GLuint hHandle);

        template<>
        inline void _Bind<gl::VBO>(GLuint hVBO) {
            glBindBuffer(GL_ARRAY_BUFFER, hVBO);
        }

        template<>
        inline void _Bind<gl::VAO>(GLuint hVAO) {
            glBindVertexArray(hVAO);
        }
    }

    template<typename T>
    inline void Bind(T const& res) {
        detail::_Bind<typename T::Resource>(res);
    }

    inline void BindElementArray(gl::VBO const& vbo) {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo);
    }

    inline void UploadArray(gl::VBO const& vbo, GLsizei size, void const* data) {
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);
    }

    inline void UploadElementArray(gl::VBO const& vbo, GLsizei size, void const* data) {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);
    }

    template<typename Shader>
    inline bool CompileShaderFromString(Shader const& shader, char const* pszSource) {
        GLint bSuccess;
        char const* aSources[1] = { pszSource };
        glShaderSource(shader, 1, aSources, NULL);
        glCompileShader(shader);
        glGetShaderiv(shader, GL_COMPILE_STATUS, &bSuccess);

        if (bSuccess == 0) {
            char pchMsgBuf[128];
            glGetShaderInfoLog(shader, 128, NULL, pchMsgBuf);
            printf("CompileShaderFromString failed: %s\n", pchMsgBuf);
        }

        return bSuccess != 0;
    }
}
