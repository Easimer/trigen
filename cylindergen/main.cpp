// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: Generate 3D mesh from a Lindenmayer-system
//

#include <cassert>
#include <optional>
#include <SDL.h>
#include "glad/glad.h"
#include <trigen/sdl_helper.h>
#include <trigen/linear_math.h>
#include "glres.h"
#include "meshbuilder.h"
#include "trunk_generator.h"

struct GL_Renderer : public sdl::Renderer {
    SDL_GLContext glctx;

    GL_Renderer() : Renderer("cylindergen", 1280, 720, SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL) {
        if (window != NULL && renderer != NULL) {
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 5);
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
            SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
            SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
            SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
            SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);

            glctx = SDL_GL_CreateContext(window);
        }
    }

    ~GL_Renderer() {
        if (glctx != NULL) {
            SDL_GL_DeleteContext(glctx);
        }
    }

    operator bool() const {
        return window && renderer && glctx;
    }

    void Present() const {
        SDL_GL_SwapWindow(window);
    }
};

static void GLMessageCallback
(GLenum src, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* lparam) {
    if (length == 0) return;
    if (severity == GL_DEBUG_SEVERITY_HIGH) {
        printf("[ gfx ] BACKEND ERROR!! '%s'\n", message);
        assert(0);
    }
#ifndef NDEBUG
    else if (severity == GL_DEBUG_SEVERITY_MEDIUM) {
        printf("[ gfx ] BACKEND WARNING: '%s'\n", message);
    } else if (severity == GL_DEBUG_SEVERITY_LOW) {
        printf("[ gfx ] backend warning: '%s'\n", message);
    } else if (severity == GL_DEBUG_SEVERITY_NOTIFICATION) {
        printf("[ gfx ] backend note: '%s'\n", message);
    }
#endif
}

void Bind(gl::VBO const& vbo) {
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
}

void BindElements(gl::VBO const& vbo) {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo);
}

void Bind(gl::VAO const& vao) {
    glBindVertexArray(vao);
}

static void UploadStaticVertices(gl::VBO const& vbo, GLsizei size, void const* data) {
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);
}

static void UploadElementArray(gl::VBO const& vbo, GLsizei size, void const* data) {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);
}

template<typename Shader>
static bool CompileShaderFromString(Shader const& shader, char const* pszSource) {
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

static gl::VAO BuildVAO(gl::VBO const& hPos) {
    gl::VAO vao;

    glBindVertexArray(vao);
    // Position buffer
    glBindBuffer(GL_ARRAY_BUFFER, hPos);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    return vao;
}

struct Element_Model {
    gl::VAO vao;
    gl::VBO vbo_vertices, vbo_elements;
    const size_t elements;
};

struct Draw_Load_Unit {
    gl::Shader_Program program;
    Element_Model mdl;
};

static Element_Model BuildModel(Mesh_Builder::Optimized_Mesh const& opt) {
    gl::VAO vao;
    gl::VBO vbo_vertices;
    gl::VBO vbo_elements;

    Bind(vao);
    UploadStaticVertices(vbo_vertices, opt.VerticesSize(), opt.vertices.data());
    UploadElementArray(vbo_elements, opt.ElementsSize(), opt.elements.data());

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    return {
        std::move(vao),
        std::move(vbo_vertices), std::move(vbo_elements),
        opt.elements.size() };
}

static void Draw(Element_Model const& elemmdl) {
    Bind(elemmdl.vao);
    glDrawElements(GL_TRIANGLES, elemmdl.elements, GL_UNSIGNED_INT, 0);
}

static void RenderLoop(GL_Renderer& r, Draw_Load_Unit const& dlu) {
    bool bExit = false;
    while (!bExit) {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            switch (ev.type) {
            case SDL_QUIT:
                bExit = true;
                break;
            }
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(dlu.program);
        Draw(dlu.mdl);

        r.Present();
    }
}

static char const* pszVertexShaderSource =
"#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"void main()\n"
"{\n"
"   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
"}";

static char const* pszFragmentShaderSource =
"#version 330 core\n"
"out vec4 FragColor;\n"
"void main() {\n"
"FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
"} ";

int main(int argc, char** argv) {
    SDL_Init(SDL_INIT_EVERYTHING);
    GL_Renderer r;
    if (r) {
        SDL_GL_SetSwapInterval(-1);
        gladLoadGLLoader(SDL_GL_GetProcAddress);

        if (glDebugMessageCallback) {
            glDebugMessageCallback(GLMessageCallback, 0);
        } else {
            printf("[ gfx ] BACKEND WARNING: no messages will be received from the driver!\n");
        }

        lm::Vector4 const vertices[] = {
            {0.5f,  0.5f, 0.0f,},
            {0.5f, -0.5f, 0.0f,},
            {-0.5f,  0.5f, 0.0f,},
            {0.5f, -0.5f, 0.0f,},
            {-0.5f, -0.5f, 0.0f,},
            {-0.5f,  0.5f, 0.0f},
        };

        lm::Vector4 const controlPoints[] = {
            {0.0f, -32.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {16.0f, 32.0f, 0.0f},
            {128.0f, 256.0f, 0.0f},
        };

        Mesh_Builder mb;
        mb.PushTriangle(vertices[0], vertices[1], vertices[2]);
        mb.PushTriangle(vertices[3], vertices[4], vertices[5]);
        auto optmesh = mb.Optimize();

        Catmull_Rom<lm::Vector4> cr(controlPoints[0], controlPoints[1], controlPoints[2], controlPoints[3]);
        auto optmesh2 = MeshFromSpline(cr);
        auto asd = BuildModel(optmesh2);

        gl::Vertex_Shader vsh;
        gl::Fragment_Shader fsh;

        CompileShaderFromString(vsh, pszVertexShaderSource);
        CompileShaderFromString(fsh, pszFragmentShaderSource);
        auto builder = gl::Shader_Program_Builder();
        auto program = builder.Attach(vsh).Attach(fsh).Link();
        if (program) {
            auto hProgram = std::move(program.value());
            // Draw_Load_Unit dlu = { std::move(hProgram), BuildModel(optmesh) };
            Draw_Load_Unit dlu = { std::move(hProgram), std::move(asd) };
            RenderLoop(r, dlu);
        } else {
            printf("Failed to link shader program: %s\n", builder.Error());
        }
    }
    SDL_Quit();
    return 0;
}