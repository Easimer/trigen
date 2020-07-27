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
#include <queue>

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
    GL_Renderer(char const* pszTitle) :
        m_width(1280), m_height(720),
        Renderer(pszTitle, 1280, 720, SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL),
        glctx(NULL),
        m_pImGuiCtx(NULL),
        m_view(glm::translate(Vec3(0.0f, 0.0f, -15.0f))),
        m_proj(glm::perspective(glm::radians(90.0f), 720.0f / 1280.0f, 0.01f, 8192.0f)),
        m_uiTimeStart(0) {
        if (window != NULL && renderer != NULL) {

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
            LoadShader("ellipsoid.vsh.glsl", "ellipsoid.fsh.glsl", m_sdf_ellipsoid_shader);

            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_LESS);
            glLineWidth(2.0f);
        }
    }

    ~GL_Renderer() {
        if (glctx != NULL) {
            ImGui_ImplOpenGL3_Shutdown();
            ImGui_ImplSDL2_Shutdown();
            SDL_GL_DeleteContext(glctx);
        }
    }

    void LoadShader(char const* pathVsh, char const* pathFsh, std::optional<gl::Shader_Program>& out) {
        auto vsh = FromFileLoadShader<GL_VERTEX_SHADER>(pathVsh);
        auto fsh = FromFileLoadShader<GL_FRAGMENT_SHADER>(pathFsh);
        if (vsh && fsh) {
            auto builder = gl::Shader_Program_Builder();
            auto program = builder.Attach(vsh.value()).Attach(fsh.value()).Link();
            if (program) {
                out = std::move(program.value());
            }
        }
    }

    virtual void set_camera(Mat4 const& view_matrix) override {
        m_view = view_matrix;
    }

    virtual void draw_points(Vec3 const* pPoints, size_t nCount, Vec3 const& vWorldPosition) override {
        if (m_point_cloud_shader.has_value()) {
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

            auto& shader = m_point_cloud_shader.value();
            glUseProgram(shader);
            
            auto const matModel = glm::translate(vWorldPosition);

            auto const locView = gl::Uniform_Location<Mat4>(shader, "matView");
            auto const locProj = gl::Uniform_Location<Mat4>(shader, "matProj");
            auto const locModel = gl::Uniform_Location<Mat4>(shader, "matModel");

            gl::SetUniformLocation(locModel , matModel);
            gl::SetUniformLocation(locView, m_view);
            gl::SetUniformLocation(locProj, m_proj);

            glDrawArrays(GL_POINTS, 0, nCount);

            glDeleteBuffers(1, &vbo);
            glDeleteVertexArrays(1, &vao);
        } else {
            printf("Can't draw: no shader!\n");
        }
    }

    void draw_ellipsoids(
        gfx::Render_Context_Supplement const& ctx,
        size_t count,
        Vec3 const* centers,
        Vec3 const* sizes,
        Quat const* rotations,
        Vec3 const& color
    ) override {
        if (m_sdf_ellipsoid_shader) {
            // Setup screen quad
            float quad[] = {
                -1,  1,
                 1,  1,
                -1, -1,
                 1, -1,
            };
            GLuint vao, vbo;
            glCreateVertexArrays(1, &vao);
            glCreateBuffers(1, &vbo);
            glBindVertexArray(vao);

            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferData(GL_ARRAY_BUFFER, 4 * 2 * sizeof(float), quad, GL_STREAM_DRAW);

            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);

            // Setup shader
            auto& shader = *m_sdf_ellipsoid_shader;
            glUseProgram(shader);

            auto const locVP = gl::Uniform_Location<Mat4>(shader, "matVP");
            auto const locInvVP = gl::Uniform_Location<Mat4>(shader, "matInvVP");
            auto const locSiz = gl::Uniform_Location<Vec3>(shader, "vSize");
            auto const locTranslation = gl::Uniform_Location<Vec3>(shader, "vTranslation");
            auto const locInvRotation = gl::Uniform_Location<Mat3>(shader, "matInvRotation");
            auto const locSun = gl::Uniform_Location<Vec3>(shader, "vSun");
            auto const locColor = gl::Uniform_Location<Vec3>(shader, "vColor");

            // NOTE(danielm): translation and rotation are not part of the MVP, they are
            // supplied separately to the GPU
            // TODO(danielm): do we really need to tho?
            auto matVP = m_proj * m_view;
            auto matInvVP = glm::inverse(matVP);
            gl::SetUniformLocation(locVP, matVP);
            gl::SetUniformLocation(locInvVP, matInvVP);

            // Set the position of the Sun
            gl::SetUniformLocation(locSun, ctx.sun ? *ctx.sun : Vec3(10, 10, 10));
            gl::SetUniformLocation(locColor, color);

            // TODO(danielm): we should render multiple objects at a time,
            // like uploading a 4-tuple of these parameters and taking their
            // union
            for (size_t i = 0; i < count; i++) {
                gl::SetUniformLocation(locTranslation, centers[i]);
                gl::SetUniformLocation(locInvRotation, Mat3(glm::conjugate(rotations[i])));
                gl::SetUniformLocation(locSiz, sizes[i]);
                glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            }

            glDeleteBuffers(1, &vbo);
            glDeleteVertexArrays(1, &vao);
        }
    }

    virtual void new_frame() override {
        // Hotload shader every frame
        LoadShader("ellipsoid.vsh.glsl", "ellipsoid.fsh.glsl", m_sdf_ellipsoid_shader);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame(window);
        ImGui::NewFrame();

        line_recycler.flip();
    }

    virtual double present() override {
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        auto const uiTimeEnd = SDL_GetPerformanceCounter();
        auto const flFrameTime = (uiTimeEnd - m_uiTimeStart) / (double)SDL_GetPerformanceFrequency();

        SDL_GL_SwapWindow(window);

        m_uiTimeStart = SDL_GetPerformanceCounter();

        SDL_Delay(0);

        return flFrameTime;
    }

    virtual bool pump_event_queue(SDL_Event& ev) override {
        auto& io = ImGui::GetIO();
        bool filtered = false;

        do {
            filtered = false;
            if (SDL_PollEvent(&ev)) {
                ImGui_ImplSDL2_ProcessEvent(&ev);

                if (io.WantCaptureKeyboard) {
                    switch (ev.type) {
                    case SDL_KEYDOWN:
                    case SDL_KEYUP:
                        filtered = true;
                        break;
                    default:
                        break;
                    }
                }
                if (io.WantCaptureMouse) {
                    switch (ev.type) {
                    case SDL_MOUSEMOTION:
                    case SDL_MOUSEBUTTONDOWN:
                    case SDL_MOUSEBUTTONUP:
                        filtered = true;
                        break;
                    }
                }
            } else {
                return false;
            }
        } while (filtered);

        return true;
    }

    struct Line {
        gl::VAO arr;
        gl::VBO buf[2];
    };

    /**
     * VAO and VBO recycler for stream drawing.
     */
    template<typename Tuple>
    struct Array_Recycler {
    public:
        /**
         * Get an unused instance of `Tuple`.
         * @param out Where the pointer to the tuple will be placed.
         * @return Handle to the instance.
         */
        size_t get(Tuple** out) {
            *out = NULL;
            if (ready_queue.empty()) {
                return make_new(out);
            } else {
                auto ret = ready_queue.front();
                ready_queue.pop();
                *out = &arrays[ret];
                return ret;
            }
        }

        /**
         * Mark a tuple instance as used and retire it.
         * @param handle An instance handle returned from get(Tuple**).
         */
        void put_back(size_t handle) {
            assert(handle < arrays.size());
            retired_queue.push(handle);
        }

        /**
         * Called after a frame ends.
         */
        void flip() {
            while (!retired_queue.empty()) {
                auto h = retired_queue.front();
                retired_queue.pop();
                ready_queue.push(h);
            }

            assert(retired_queue.size() == 0);
            assert(ready_queue.size() == arrays.size());
        }
    protected:
        size_t make_new(Tuple** out) {
            auto ret = arrays.size();
            arrays.push_back(Tuple());
            *out = &arrays.back();
            return ret;
        }
    private:
        std::queue<size_t> ready_queue;
        std::queue<size_t> retired_queue;
        std::vector<Tuple> arrays;
    };

    Array_Recycler<Line> line_recycler;

    virtual void draw_lines(
        Vec3 const* pEndpoints,
        size_t nLineCount,
        Vec3 const& vWorldPosition,
        Vec3 const& vStartColor,
        Vec3 const& vEndColor
    ) override {
        if (nLineCount == 0) return;

        Line* l;
        auto l_h = line_recycler.get(&l);

        glBindVertexArray(l->arr);

        auto zero_one = std::make_unique<float[]>(nLineCount * 2);
        auto zero_one_p = zero_one.get();
        for (unsigned i = 0; i < nLineCount * 2; i++) {
            zero_one_p[i] = ((i & 1) == 0) ? 0.0f : 1.0f;
        }

        auto const size = nLineCount * 2 * 3 * sizeof(float);
        auto const data = pEndpoints;

        glBindBuffer(GL_ARRAY_BUFFER, l->buf[0]);
        glBufferData(GL_ARRAY_BUFFER, size, data, GL_STREAM_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);

        glBindBuffer(GL_ARRAY_BUFFER, l->buf[1]);
        glBufferData(GL_ARRAY_BUFFER, nLineCount * 2 * sizeof(float), zero_one_p, GL_STREAM_DRAW);
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 1 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);

        if (m_line_shader.has_value()) {
            auto& shader = m_line_shader.value();
            glUseProgram(shader);
            auto locMVP = gl::Uniform_Location<Mat4>(shader, "matMVP");
            auto locColor0 = gl::Uniform_Location<Vec3>(shader, "vColor0");
            auto locColor1 = gl::Uniform_Location<Vec3>(shader, "vColor1");
            auto matModel = glm::translate(vWorldPosition);
            auto matMVP = m_proj * m_view * matModel;
            gl::SetUniformLocation(locMVP, matMVP);
            gl::SetUniformLocation(locColor0, vStartColor);
            gl::SetUniformLocation(locColor1, vEndColor);

            glDrawArrays(GL_LINES, 0, 2 * nLineCount);

            line_recycler.put_back(l_h);
        } else {
            printf("Can't draw: no shader!\n");
        }
    }

    virtual void change_resolution(unsigned* inout_width, unsigned* inout_height) override {
        assert(inout_width != NULL);
        assert(inout_height != NULL);

        m_width = *inout_width;
        m_height = *inout_height;
        SDL_SetWindowSize(window, m_width, m_height);

        m_proj = glm::perspective(glm::radians(90.0f), (*inout_height) / (float)(*inout_width), 0.01f, 1000.0f);
    }

    virtual void get_resolution(unsigned* out_width, unsigned* out_height) override {
        *out_width = m_width;
        *out_height = m_height;
    }

private:
    unsigned m_width, m_height;
    SDL_GLContext glctx;
    ImGuiContext* m_pImGuiCtx;
    Mat4 m_view;
    Mat4 m_proj;
    decltype(SDL_GetPerformanceCounter()) m_uiTimeStart;
    std::optional<gl::Shader_Program> m_point_cloud_shader;
    std::optional<gl::Shader_Program> m_line_shader;
    std::optional<gl::Shader_Program> m_sdf_ellipsoid_shader;
};

gfx::IRenderer* gfx::make_renderer(gfx::Renderer_Config const& cfg) {
    SDL_Init(SDL_INIT_EVERYTHING);
    return new GL_Renderer(cfg.title);
}

void gfx::destroy_renderer(gfx::IRenderer* r) {
    auto gl_r = static_cast<GL_Renderer*>(r);
    delete gl_r;
    SDL_Quit();
}
