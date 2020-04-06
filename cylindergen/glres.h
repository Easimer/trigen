// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: OpenGL resource management
//

#pragma once
#include "glad/glad.h"
#include <utility>
#include <optional>

namespace gl {
    template<typename T>
    using Optional = std::optional<T>;

    template<typename Handle, typename Allocator>
    class Managed_Resource {
    public:
        using Handle_Type = Handle;
        using Allocator_Type = Allocator;

        Managed_Resource() {
            Allocator::Allocate(&handle);
        }

        ~Managed_Resource() {
            Allocator::Deallocate(&handle);
        }

        Managed_Resource(Managed_Resource const&) = delete;
        Managed_Resource(Managed_Resource&& other) : handle(0) {
            std::swap(handle, other.handle);
        }

        void operator=(Managed_Resource const&) = delete;
        Managed_Resource& operator=(Managed_Resource&& other) {
            Allocator::Deallocate(&handle);
            handle = Handle();
            std::swap(handle, other.handle);
            return *this;
        }

        operator Handle () const {
            return handle;
        }
    private:
    private:
        Handle handle;
    };

    template<typename Resource>
    class Weak_Resource_Reference {
    public:
        using Handle_Type = typename Resource::Handle_Type;
        Weak_Resource_Reference(Resource const& res) {
            handle = res;
        }
    private:
        Handle_Type handle;
    };

    struct VBO_Allocator {
        static void Allocate(GLuint* pHandle) {
            glCreateBuffers(1, pHandle);
        }

        static void Deallocate(GLuint* pHandle) {
            glDeleteBuffers(1, pHandle);
        }
    };

    struct VAO_Allocator {
        static void Allocate(GLuint* pHandle) {
            glGenVertexArrays(1, pHandle);
        }

        static void Deallocate(GLuint* pHandle) {
            glDeleteVertexArrays(1, pHandle);
        }
    };

    template<GLenum shaderType>
    struct Shader_Allocator {
        static GLenum const ShaderType = shaderType;
        static void Allocate(GLuint* pHandle) {
            *pHandle = glCreateShader(shaderType);
        }

        static void Deallocate(GLuint* pHandle) {
            glDeleteShader(*pHandle);
        }
    };

    struct Shader_Program_Allocator {
        static void Allocate(GLuint* pHandle) {
            *pHandle = glCreateProgram();
        }

        static void Deallocate(GLuint* pHandle) {
            glDeleteProgram(*pHandle);
        }
    };

    using VBO = Managed_Resource<GLuint, VBO_Allocator>;
    using VAO = Managed_Resource<GLuint, VAO_Allocator>;
    using Vertex_Shader = Managed_Resource<GLuint, Shader_Allocator<GL_VERTEX_SHADER>>;
    using Fragment_Shader = Managed_Resource<GLuint, Shader_Allocator<GL_FRAGMENT_SHADER>>;
    using Shader_Program = Managed_Resource<GLuint, Shader_Program_Allocator>;

    class Shader_Program_Builder {
    public:
        Shader_Program_Builder& Attach(Vertex_Shader const& vsh) {
            glAttachShader(program, vsh);
            return *this;
        }

        Shader_Program_Builder& Attach(Fragment_Shader const& fsh) {
            glAttachShader(program, fsh);
            return *this;
        }

        Optional<Shader_Program> Link() {
            glLinkProgram(program);
            GLint bSuccess;
            glGetProgramiv(program, GL_LINK_STATUS, &bSuccess);
            if (bSuccess != 0) {
                return { std::move(program) };
            } else {
                glGetProgramInfoLog(program, 256, NULL, errorMsg);
            }

            return {};
        }

        char const* Error() const { return errorMsg; }
    private:
        Shader_Program program;
        char errorMsg[256] = { 0 };
    };
}
