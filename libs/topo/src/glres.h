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
        using Resource = Managed_Resource<Handle, Allocator>;

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

    template<typename R>
    class Weak_Resource_Reference {
    public:
        using Resource = R;
        using Handle_Type = typename Resource::Handle_Type;
        Weak_Resource_Reference(Resource const& res) {
            handle = res;
        }

        operator Handle_Type() const {
            return handle;
        }
    private:
        Handle_Type handle;
    };

    struct VBO_Allocator {
        static void Allocate(GLuint* pHandle) {
            glGenBuffers(1, pHandle);
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

    struct Texture_Allocator {
        static void Allocate(GLuint* pHandle) {
            glGenTextures(1, pHandle);
        }

        static void Deallocate(GLuint* pHandle) {
            glDeleteTextures(1, pHandle);
        }
    };

    struct Framebuffer_Allocator {
        static void Allocate(GLuint* pHandle) {
            glGenFramebuffers(1, pHandle);
        }

        static void Deallocate(GLuint* pHandle) {
            glDeleteFramebuffers(1, pHandle);
        }
    };

    struct Renderbuffer_Allocator {
        static void Allocate(GLuint* pHandle) {
            glGenRenderbuffers(1, pHandle);
        }

        static void Deallocate(GLuint* pHandle) {
            glDeleteRenderbuffers(1, pHandle);
        }
    };

    using VBO = Managed_Resource<GLuint, VBO_Allocator>;
    using VAO = Managed_Resource<GLuint, VAO_Allocator>;
    template<GLenum kType>
    using Shader = Managed_Resource<GLuint, Shader_Allocator<kType>>;
    using Vertex_Shader = Shader<GL_VERTEX_SHADER>;
    using Fragment_Shader = Shader<GL_FRAGMENT_SHADER>;
    using Compute_Shader = Shader<GL_COMPUTE_SHADER>;
    using Shader_Program = Managed_Resource<GLuint, Shader_Program_Allocator>;
    using Texture = Managed_Resource<GLuint, Texture_Allocator>;
    using Framebuffer = Managed_Resource<GLuint, Framebuffer_Allocator>;
    using Renderbuffer = Managed_Resource<GLuint, Renderbuffer_Allocator>;

    template<typename T>
    struct Uniform_Location {
    public:
        Uniform_Location(Shader_Program const& hProgram, char const* pszName) {
            loc = glGetUniformLocation(hProgram, pszName);
        }

        Uniform_Location(Weak_Resource_Reference<Shader_Program> const& hProgram, char const* pszName) {
            loc = glGetUniformLocation(hProgram, pszName);
        }

        Uniform_Location()
            : loc(-1) { }

        explicit Uniform_Location(GLint loc) : loc(loc) {}

        operator GLint() const { return loc; }
    private:
        GLint loc;
    };

    template<typename T>
    inline void SetUniformLocation(Uniform_Location<T> const&, T const&) = delete;

    template<typename T>
    inline void SetUniformLocationArray(Uniform_Location<T*> const&, T const*, unsigned count) = delete;

    template<>
    inline void SetUniformLocation<GLint>(Uniform_Location<GLint> const &uiLoc, GLint const& id) {
        glUniform1i(uiLoc, id);
    }

    template<>
    inline void SetUniformLocation<GLuint>(Uniform_Location<GLuint> const &uiLoc, GLuint const& id) {
        glUniform1ui(uiLoc, id);
    }

#ifdef GLRES_GLM
    template<>
    inline void SetUniformLocation<glm::mat4>(Uniform_Location<glm::mat4> const& uiLoc, glm::mat4 const& matMatrix) {
        glUniformMatrix4fv(uiLoc, 1, GL_FALSE, glm::value_ptr(matMatrix));
    }

    template<>
    inline void SetUniformLocation<glm::mat3>(Uniform_Location<glm::mat3> const& uiLoc, glm::mat3 const& matMatrix) {
        glUniformMatrix3fv(uiLoc, 1, GL_FALSE, glm::value_ptr(matMatrix));
    }

    template<>
    inline void SetUniformLocation<glm::vec3>(Uniform_Location<glm::vec3> const& uiLoc, glm::vec3 const& vVec) {
        glUniform3fv(uiLoc, 1, glm::value_ptr(vVec));
    }

    template<>
    inline void SetUniformLocation<glm::vec4>(Uniform_Location<glm::vec4> const& uiLoc, glm::vec4 const& vVec) {
        glUniform4fv(uiLoc, 1, glm::value_ptr(vVec));
    }

    template<>
    inline void SetUniformLocationArray<glm::vec3>(Uniform_Location<glm::vec3*> const& uiLoc, glm::vec3 const* vVec, unsigned count) {
        glUniform3fv(uiLoc, count, (float const*)vVec);
    }

    template<>
    inline void SetUniformLocationArray<glm::mat3>(Uniform_Location<glm::mat3*> const& uiLoc, glm::mat3 const* matMatrix, unsigned count) {
        glUniformMatrix3fv(uiLoc, count, GL_FALSE, (float*)matMatrix);
    }

    template<>
    inline void SetUniformLocationArray<glm::vec4>(Uniform_Location<glm::vec4*> const& uiLoc, glm::vec4 const* vVec, unsigned count) {
        glUniform4fv(uiLoc, count, (float const*)vVec);
    }

    template<>
    inline void SetUniformLocationArray<glm::mat4>(Uniform_Location<glm::mat4*> const& uiLoc, glm::mat4 const* matMatrix, unsigned count) {
        glUniformMatrix4fv(uiLoc, count, GL_FALSE, (float*)matMatrix);
    }
#endif
}