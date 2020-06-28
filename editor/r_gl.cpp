// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: editor entry point
//

#include "stdafx.h"
#include "r_renderer.h"
#include "r_queue.h"
#include <trigen/sdl_helper.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_sdl.h>

#define GLRES_GLM
#include <glm/gtc/type_ptr.hpp>
#include <trigen/glres.h>
#include <optional>
#include <fstream>
#include <sstream>

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

template<GLenum kType>
static std::optional<gl::Shader<kType>> FromFileLoadShader(char const* pszPath) {
    gl::Shader<kType> shader; 

    std::ifstream f(pszPath);
    if (f) {
        std::stringstream ss;
        ss << f.rdbuf();
        if (CompileShaderFromString(shader, ss.str().c_str())) {
            return shader;
        }
    }

    return {};
}

class GL_Renderer : public sdl::Renderer, public gfx::IRenderer {
public:
    GL_Renderer() :
        Renderer("editor", 1280, 720, SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL),
        glctx(NULL),
        m_pImGuiCtx(NULL),
        m_view(glm::translate(Vec3(0.0f, 0.0f, -15.0f))),
        m_uiTimeStart(0) {
        if (window != NULL && renderer != NULL) {
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 5);
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
            SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
            SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
            SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
            SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);

            glctx = SDL_GL_CreateContext(window);

            SDL_GL_SetSwapInterval(-1);
            gladLoadGLLoader(SDL_GL_GetProcAddress);

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            if (glDebugMessageCallback) {
                glDebugMessageCallback(GLMessageCallback, 0);
            } else {
                printf("[ gfx ] BACKEND WARNING: no messages will be received from the driver!\n");
            }

            IMGUI_CHECKVERSION();
            m_pImGuiCtx = ImGui::CreateContext();
            ImGui::GetIO();
            ImGui::StyleColorsDark();
            ImGui_ImplSDL2_InitForOpenGL(window, glctx);
            ImGui_ImplOpenGL3_Init("#version 130");

            m_uiTimeStart = SDL_GetPerformanceCounter();

            {
                auto vsh = FromFileLoadShader<GL_VERTEX_SHADER>("points.vsh.glsl");
                auto fsh = FromFileLoadShader<GL_FRAGMENT_SHADER>("points.fsh.glsl");
                auto builder = gl::Shader_Program_Builder();
                auto program = builder.Attach(vsh.value()).Attach(fsh.value()).Link();
                m_point_cloud_shader = std::move(program.value());
            }
            {
                auto vsh = FromFileLoadShader<GL_VERTEX_SHADER>("lines.vsh.glsl");
                auto fsh = FromFileLoadShader<GL_FRAGMENT_SHADER>("lines.fsh.glsl");
                auto builder = gl::Shader_Program_Builder();
                auto program = builder.Attach(vsh.value()).Attach(fsh.value()).Link();
                m_line_shader = std::move(program.value());
            }
        }
    }

    ~GL_Renderer() {
        if (glctx != NULL) {
            ImGui_ImplOpenGL3_Shutdown();
            ImGui_ImplSDL2_Shutdown();
            SDL_GL_DeleteContext(glctx);
        }
    }

    virtual void set_camera(Mat4 const& view_matrix) override {
        m_view = view_matrix;
    }

    virtual void draw_points(Vec3 const* pPoints, size_t nCount, Vec3 const& vWorldPosition) override {
        GLuint vao, vbo;
        glCreateVertexArrays(1, &vao);
        glCreateBuffers(1, &vbo);
        glBindVertexArray(vao);

        auto const size = nCount * 3 * sizeof(float);
        auto const data = pPoints;
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, size, data, GL_STREAM_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);

        if (m_point_cloud_shader.has_value()) {
            auto& shader = m_point_cloud_shader.value();
            glUseProgram(shader);
            auto locView = gl::Uniform_Location<Mat4>(shader, "matView");
            auto locProj = gl::Uniform_Location<Mat4>(shader, "matProj");
            auto locModel = gl::Uniform_Location<Mat4>(shader, "matModel");
            // auto matMVP = m_view * glm::perspective(1.57079633f, 720.0f / 1280.0f, 0.01f, 1000.0f);
            auto matProj = glm::perspective(1.57079633f, 1280.0f / 720.0f, 0.01f, 1000.0f);
            auto matModel = Mat4(1.0f);
            gl::SetUniformLocation(locModel , matModel);
            gl::SetUniformLocation(locView, m_view);
            gl::SetUniformLocation(locProj, matProj);

            glDrawArrays(GL_POINTS, 0, nCount);

            glDeleteBuffers(1, &vbo);
            glDeleteVertexArrays(1, &vao);
        } else {
            printf("Can't draw: no shader!\n");
        }
    }

    virtual void new_frame() override {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame(window);
        ImGui::NewFrame();

    }

    virtual double present() override {
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        auto const uiTimeEnd = SDL_GetPerformanceCounter();
        auto const flFrameTime = (uiTimeEnd - m_uiTimeStart) / (double)SDL_GetPerformanceFrequency();

        SDL_GL_SwapWindow(window);

        m_uiTimeStart = SDL_GetPerformanceCounter();

        return flFrameTime;
    }

    virtual bool pump_event_queue(SDL_Event& ev) override {
        bool ret = SDL_PollEvent(&ev);

        if (ret) {
            ImGui_ImplSDL2_ProcessEvent(&ev);
        }

        return ret;
    }

    virtual void draw_lines(
        Vec3 const* pEndpoints,
        size_t nLineCount,
        Vec3 const& vWorldPosition,
        Vec3 const& vStartColor,
        Vec3 const& vEndColor
    ) override {
        if (nLineCount == 0) return;

        GLuint vao;
        GLuint vbo[2];
        glCreateVertexArrays(1, &vao);
        glCreateBuffers(2, vbo);
        glBindVertexArray(vao);

        auto zero_one = std::make_unique<float[]>(nLineCount * 2);
        auto zero_one_p = zero_one.get();
        for (unsigned i = 0; i < nLineCount * 2; i++) {
            zero_one_p[i] = ((i & 1) == 0) ? 0.0f : 1.0f;
        }

        auto const size = nLineCount * 2 * 3 * sizeof(float);
        auto const data = pEndpoints;

        glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
        glBufferData(GL_ARRAY_BUFFER, size, data, GL_STREAM_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);

        glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
        glBufferData(GL_ARRAY_BUFFER, nLineCount * 2 * sizeof(float), zero_one_p, GL_STREAM_DRAW);
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 1 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);

        if (m_line_shader.has_value()) {
            auto& shader = m_line_shader.value();
            glUseProgram(shader);
            auto locView = gl::Uniform_Location<Mat4>(shader, "matView");
            auto locProj = gl::Uniform_Location<Mat4>(shader, "matProj");
            auto locModel = gl::Uniform_Location<Mat4>(shader, "matModel");
            auto locColor0 = gl::Uniform_Location<Vec3>(shader, "vColor0");
            auto locColor1 = gl::Uniform_Location<Vec3>(shader, "vColor1");
            auto matProj = glm::perspective(1.57079633f, 1280.0f / 720.0f, 0.01f, 1000.0f);
            auto matModel = Mat4(1.0f);
            gl::SetUniformLocation(locModel , matModel);
            gl::SetUniformLocation(locView, m_view);
            gl::SetUniformLocation(locProj, matProj);
            gl::SetUniformLocation(locColor0, vStartColor);
            gl::SetUniformLocation(locColor1, vEndColor);

            glDrawArrays(GL_LINES, 0, 2 * nLineCount);

            glDeleteBuffers(2, vbo);
            glDeleteVertexArrays(1, &vao);
        } else {
            printf("Can't draw: no shader!\n");
        }
    }

private:
    SDL_GLContext glctx;
    ImGuiContext* m_pImGuiCtx;
    Mat4 m_view;
    decltype(SDL_GetPerformanceCounter()) m_uiTimeStart;
    std::optional<gl::Shader_Program> m_point_cloud_shader;
    std::optional<gl::Shader_Program> m_line_shader;
};

gfx::IRenderer* gfx::make_renderer() {
    SDL_Init(SDL_INIT_EVERYTHING);
    return new GL_Renderer();
}

void gfx::destroy_renderer(gfx::IRenderer* r) {
    auto gl_r = static_cast<GL_Renderer*>(r);
    delete gl_r;
    SDL_Quit();
}
