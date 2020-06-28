// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: editor entry point
//

#include "stdafx.h"
#include "application.h"
/*
#include <trigen/sdl_helper.h>
#include <trigen/GL.h>
#include <trigen/lindenmayer.h>

#include <sstream>
#include <fstream>

#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_sdl.h>

struct GL_Renderer : public sdl::Renderer {
    SDL_GLContext glctx;

    GL_Renderer() : Renderer("editor", 1280, 720, SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL), glctx(NULL) {
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

struct Element_Model {
    gl::VAO vao;
    gl::VBO vbo_vertices, vbo_elements;
    size_t elements;
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

#define PUSH_GUI_CTX() auto __guictx = ImGui::GetCurrentContext()
#define SET_GUI_CTX(ctx) ImGui::SetCurrentContext(ctx)
#define POP_GUI_CTX() ImGui::SetCurrentContext(__guictx);

enum Application_Result {
    k_nApplication_Result_OK = 0,
    k_nApplication_Result_Quit = 1,
    k_nApplication_Result_GeneralFailure = -1,
};

struct Application_Data {
    gl::Shader_Program program;
};

static Application_Data* gpAppData = NULL;

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

static Application_Result OnPreFrame() {
    if (gpAppData == NULL) {
        gpAppData = new Application_Data;
        auto vsh = FromFileLoadShader<GL_VERTEX_SHADER>("generic.vsh.glsl");
        auto fsh = FromFileLoadShader<GL_FRAGMENT_SHADER>("generic.fsh.glsl");
        auto builder = gl::Shader_Program_Builder();
        auto program = builder.Attach(vsh.value()).Attach(fsh.value()).Link();
        if (program) {
            gpAppData->program = std::move(program.value());
        }
    }

    return k_nApplication_Result_OK;
}

static Application_Result OnInput(SDL_Event const& ev) {
    return k_nApplication_Result_OK;
}

static Application_Result OnDraw(rq::Render_Queue* pQueue) {
    if (ImGui::Begin("Test")) {
        if (ImGui::Button("Hello")) {
            printf("World!\n");
        }

        gpAppData->ts.Edit(ImGui::GetCurrentContext());
    }
    ImGui::End();
    Tree_Renderer::Render(pQueue, gpAppData->tree, lm::Vector4(), gpAppData->program);
    return k_nApplication_Result_OK;
}

static Application_Result OnPostFrame() {
    return k_nApplication_Result_OK;
}

int old_main(int argc, char** argv) {
    printf("editor v0.0.1\n");
    printf("Initializing SDL2\n");
    SDL_Init(SDL_INIT_EVERYTHING);
    printf("Initializing GL_Renderer\n");
    GL_Renderer R;
    bool bExit = false;
    ImGuiContext* pImGuiCtx = NULL;

    if (R) {
        SDL_GL_SetSwapInterval(-1);
        printf("Initializing GLAD\n");
        gladLoadGLLoader(SDL_GL_GetProcAddress);

        if (glDebugMessageCallback) {
            glDebugMessageCallback(GLMessageCallback, 0);
        } else {
            printf("[ gfx ] BACKEND WARNING: no messages will be received from the driver!\n");
        }

        printf("Initializing ImGUI\n");
        IMGUI_CHECKVERSION();
        pImGuiCtx = ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;

        ImGui::StyleColorsDark();

        ImGui_ImplSDL2_InitForOpenGL(R, (void*)R.glctx);
        ImGui_ImplOpenGL3_Init("#version 130");

        rq::Render_Queue rq;
        double flFrameTime = 0;
        Application_Result res;
        SDL_Event ev;
#define CHECK_QUIT() if(res != k_nApplication_Result_OK) bExit = true

        while (!bExit) {
            rq.Clear();
            auto const uiTimeStart = SDL_GetPerformanceCounter();

            res = OnPreFrame();
            CHECK_QUIT();

            while (SDL_PollEvent(&ev)) {
                ImGui_ImplSDL2_ProcessEvent(&ev);
                switch (ev.type) {
                case SDL_QUIT:
                {
                    bExit = true;
                    break;
                }
                case SDL_KEYDOWN:
                case SDL_KEYUP:
                case SDL_MOUSEBUTTONDOWN:
                case SDL_MOUSEBUTTONUP:
                case SDL_MOUSEMOTION:
                case SDL_MOUSEWHEEL:
                {
                    res = OnInput(ev);
                    CHECK_QUIT();
                    break;
                }
                }
            }

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplSDL2_NewFrame(R);
            ImGui::NewFrame();

            res = OnDraw(&rq);
            CHECK_QUIT();

            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            res = OnPostFrame();
            CHECK_QUIT();

            auto const uiTimeEnd = SDL_GetPerformanceCounter();
            flFrameTime = (uiTimeEnd - uiTimeStart) / (double)SDL_GetPerformanceFrequency();
            R.Present();
        }

        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplSDL2_Shutdown();
    }

    SDL_Quit();
    return 0;
}
*/

int main(int argc, char** argv) {
    printf("editor v0.0.1\n");

    app_main_loop();

    return 0;
}
