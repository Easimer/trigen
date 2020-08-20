// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: Generate 3D mesh from a Lindenmayer-system
//

#include "stdafx.h"
#include <cassert>
#include <ctime>
#include <optional>
#include <fstream>
#include <sstream>
#include <stack>
#include <SDL.h>
#include "glad/glad.h"
#include <trigen/sdl_helper.h>
#include <trigen/linear_math.h>
#include "general.h"
#include <trigen/glres.h>
#include <trigen/meshbuilder.h>
#include <trigen/future_union_mesh.h>
#include <trigen/lindenmayer.h>
#include <trigen/profiler.h>

#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_sdl.h>
#include <queue>
#include <map>

template<typename R>
bool is_ready(std::future<R> const& f) {
    return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

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

template<typename T>
void _Bind(GLuint hHandle);

template<>
void _Bind<gl::VBO>(GLuint hVBO) {
    glBindBuffer(GL_ARRAY_BUFFER, hVBO);
}

template<>
void _Bind<gl::VAO>(GLuint hVAO) {
    glBindVertexArray(hVAO);
}

template<typename T>
void Bind(T const& res) {
    _Bind<typename T::Resource>(res);
}

void BindElements(gl::VBO const& vbo) {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo);
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
    size_t elements;
};

struct Draw_Load_Unit {
    gl::Shader_Program program;
    Element_Model mdl;
};

struct Cmd_Draw_Element_Model {
    gl::Weak_Resource_Reference<gl::VAO> vao;
    gl::Weak_Resource_Reference<gl::VBO> vbo_vertices, vbo_elements;
    const size_t elements;
    lm::Vector4 position;

    Cmd_Draw_Element_Model(lm::Vector4 const& pos, Element_Model const& emdl)
        : vao(emdl.vao), vbo_vertices(emdl.vbo_vertices), vbo_elements(emdl.vbo_elements),
        elements(emdl.elements), position(pos) {}
};

struct Cmd_Change_Program {
    gl::Weak_Resource_Reference<gl::Shader_Program> program;
};

using Cmd_Draw = std::variant<
    Cmd_Draw_Element_Model,
    Cmd_Change_Program
>;

using Draw_Queue = std::vector<Cmd_Draw>;

struct Camera_State {
    lm::Vector4 position;
    lm::Vector4 euler_rotation;
    lm::Matrix4 mvp;
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

struct Renderer_State {
    bool bExit = false;
    Camera_State cam;
    lm::Matrix4 matProj, matInvProj;
    float flZoom = 1.0f;
};

class Render_Pass {
public:
    Render_Pass(GL_Renderer& r, Renderer_State& state)
        : r(r), state(state) {
        auto const matView =
            lm::Scale(1.0f / state.flZoom) *
            lm::RotationY(state.cam.euler_rotation[1]) *
            lm::Translation(-state.cam.position);
        matVP = matView * state.matProj;
    }

    void operator()(Cmd_Change_Program const& chprog) {
        locMVP = gl::Uniform_Location<lm::Matrix4>(chprog.program, "matMVP");
        glUseProgram(chprog.program);
    }

    void operator()(Cmd_Draw_Element_Model const& draw) {
        auto const matMVP = lm::Translation(draw.position) * matVP;
        gl::SetUniformLocation(locMVP.value(), matMVP);
        Bind(draw.vao);
        glDrawElements(GL_TRIANGLES, draw.elements, GL_UNSIGNED_INT, 0);
    }
private:
    GL_Renderer& r;
    Renderer_State& state;

    std::optional<gl::Uniform_Location<lm::Matrix4>> locMVP;
    lm::Matrix4 matVP;
};

static void ProcessEvents(GL_Renderer& r, Renderer_State& state) {
    SDL_Event ev;

    while (SDL_PollEvent(&ev)) {
        ImGui_ImplSDL2_ProcessEvent(&ev);
        switch (ev.type) {
            case SDL_QUIT: {
                state.bExit = true;
                break;
            }
            case SDL_MOUSEMOTION: {
                if (ev.motion.state & SDL_BUTTON_RMASK) {
                    state.cam.euler_rotation =
                        state.cam.euler_rotation + lm::Vector4(0, ev.motion.xrel / 32.0f, 0);
                }
                break;
            }
            case SDL_MOUSEWHEEL: {
                state.flZoom += -ev.wheel.y;
                break;
            }
        }
    }

    if (state.flZoom < 1.0f) { state.flZoom = 1.0f; }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL2_NewFrame(r);
    ImGui::NewFrame();

}

static void RenderPass(GL_Renderer& r, Renderer_State& state, Draw_Queue const& dq) {
    Render_Pass visitor(r, state);

    for (auto const& cmd : dq) {
        std::visit(visitor, cmd);
    }
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

static float randf() {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

int main(int argc, char** argv) {
    SDL_Init(SDL_INIT_EVERYTHING);
    GL_Renderer r;

    Lindenmayer::System::Alphabet const alphabet = {
        {'[', {Lindenmayer::Op::Push}},
        {']', {Lindenmayer::Op::Pop}},
        {'-', {Lindenmayer::Op::Pitch_Neg}},
        {'+', {Lindenmayer::Op::Pitch_Pos}},
        {'F', {Lindenmayer::Op::Forward}},
        {'X', {}},
    };

    Lindenmayer::System::Rule_Set const rules = {
        {'X', "F+[[X]-X]-F[-FX]+X"},
        {'F', "FF"},
    };

    Lindenmayer::System sys("X", alphabet, rules);
    Lindenmayer::Parameters params(64.0f, 0.436332313f, 0.436332313f, 0.436332313f);
    auto tree = Lindenmayer::Execute(sys.Iterate(4), params);

    if (r) {
        SDL_GL_SetSwapInterval(-1);
        gladLoadGLLoader(SDL_GL_GetProcAddress);

        if (glDebugMessageCallback) {
            glDebugMessageCallback(GLMessageCallback, 0);
        } else {
            printf("[ gfx ] BACKEND WARNING: no messages will be received from the driver!\n");
        }

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;

        ImGui::StyleColorsDark();

        ImGui_ImplSDL2_InitForOpenGL(r, (void*)r.glctx);
        ImGui_ImplOpenGL3_Init("#version 130");

        srand(time(NULL));
        std::future<Mesh_Builder::Optimized_Mesh> meshTree;
        std::optional<Element_Model> mdlTree;

        size_t unCurrentIterationCount = 0;
        size_t unDesiredIterationCount = 2;

        auto vsh = FromFileLoadShader<GL_VERTEX_SHADER>("generic.vsh.glsl");
        auto fsh = FromFileLoadShader<GL_FRAGMENT_SHADER>("generic.fsh.glsl");

        if (vsh && fsh) {
            auto builder = gl::Shader_Program_Builder();
            auto program = builder.Attach(vsh.value()).Attach(fsh.value()).Link();
            if (program) {
                auto hProgram = std::move(program.value());
                Renderer_State state;
                lm::Perspective(state.matProj, state.matInvProj, r.width, r.height, 1.57079633f, 0.01f, 1000.0f);
                state.cam.position = lm::Vector4(0, 32, 128);
                float flFrameTime = 0;
                while (!state.bExit) {
                    Draw_Queue dq;
                    auto const uiTimeStart = SDL_GetPerformanceCounter();

                    ProcessEvents(r, state);

                    if (unCurrentIterationCount != unDesiredIterationCount) {
                        tree = Lindenmayer::Execute(sys.Iterate(unDesiredIterationCount), params);
                        mdlTree.reset();
                        meshTree = std::async(std::launch::async, [=]() {
                            return ProcessTree(tree, [](auto i, auto, auto, auto, auto, auto) -> float { return 4.0f * powf(0.99f, i + 0); });
                        });
                        unCurrentIterationCount = unDesiredIterationCount;
                    }

                    if (ImGui::Begin("Config")) {
                        ImGui::Text("Frame time: %f ms\n", flFrameTime * 1000);
                        int nDesiredIterationCount = unDesiredIterationCount;
                        ImGui::InputInt("Iteration count", &nDesiredIterationCount);
                        if (nDesiredIterationCount > 0) {
                            unDesiredIterationCount = (size_t)nDesiredIterationCount;
                        }
                        if (meshTree.valid() && !is_ready(meshTree)) {
                            ImGui::Text("Generating mesh...");
                        }
                    }
                    ImGui::End();

                    Cmd_Change_Program chprog{hProgram};
                    dq.push_back(chprog);
                    if (mdlTree.has_value()) {
                        Cmd_Draw_Element_Model draw(lm::Vector4(), mdlTree.value());
                        dq.push_back(draw);
                    } else {
                        if (meshTree.valid() && is_ready(meshTree)) {
                            mdlTree = BuildModel(meshTree.get());
                        }
                    }
                    RenderPass(r, state, dq);

                    ImGui::Render();
                    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
                    auto const uiTimeEnd = SDL_GetPerformanceCounter();
                    flFrameTime = (uiTimeEnd - uiTimeStart) / (double)SDL_GetPerformanceFrequency();
                    r.Present();
                }

            } else {
                printf("Failed to link shader program: %s\n", builder.Error());
            }
        } else {
            printf("Failed to load the generic shaders!\n");
        }

        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplSDL2_Shutdown();
    }
    SDL_Quit();
    return 0;
}