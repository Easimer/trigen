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

#define SDF_BATCH_SIZE_ORDER (5)
#define SDF_BATCH_SIZE (1 << SDF_BATCH_SIZE_ORDER)

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

struct Shader_Define {
    std::string key;
    std::string value;
};

using Shader_Define_List = std::vector<Shader_Define>;

template<typename Shader>
static bool CompileShaderFromString(Shader const& shader, char const* pszSource, char const* pszPath, Shader_Define_List const& defines) {
    GLint bSuccess;
    std::vector<std::string> defines_fmt;
    std::vector<char const*> sources;

    bool is_intel_gpu = false;

    // Detect open-source Intel drivers
    is_intel_gpu = (strcmp((char*)glGetString(GL_VENDOR), "Intel Open Source Technology Center") == 0);

    char const* pszVersion = "#version 330 core\n";
    char const* pszLineReset = "#line -1\n";

    if (is_intel_gpu) {
        pszVersion = "#version 130\n";
    }

    sources.push_back(pszVersion);

    if (is_intel_gpu) {
        sources.push_back("#define VAO_LAYOUT(i)\n");
    }

    for (auto& def : defines) {
        char buf[64];
        snprintf(buf, 63, "#define %s %s\n", def.key.c_str(), def.value.c_str());
        defines_fmt.push_back(std::string((char const*)buf));
        sources.push_back(defines_fmt.back().c_str());
    }

    if (!is_intel_gpu) {
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
        } else {
            printf("Compilation of shader '%s' has failed:\n%s\n", pszPath, pchMsgBuf);
        }
    }

    return bSuccess != 0;
}

template<GLenum kType>
static std::optional<gl::Shader<kType>> FromFileLoadShader(char const* pszPath, Shader_Define_List const& defines) {
    gl::Shader<kType> shader; 

    std::ifstream f(pszPath);
    if (f) {
        std::stringstream ss;
        ss << f.rdbuf();
        if (CompileShaderFromString(shader, ss.str().c_str(), pszPath, defines)) {
            return shader;
        }
    }

    return {};
}

template<GLenum kType>
static std::optional<gl::Shader<kType>> FromFileLoadShader(char const* pszPath) {
    Shader_Define_List x;

    return FromFileLoadShader<kType>(pszPath, x);
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

            Shader_Define_List defines = { {"BATCH_SIZE", ""} };
            for (int order = 0; order < SDF_BATCH_SIZE_ORDER + 1; order++) {
                char buf[64];
                snprintf(buf, 63, "%d", 1 << order);
                defines[0].value = (char const*)buf;
                LoadShader("ellipsoid.vsh.glsl", "ellipsoid.fsh.glsl", defines, m_sdf_ellipsoid_batch[order]);
            }

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

    void LoadShader(char const* pathVsh, char const* pathFsh, Shader_Define_List const& defines, std::optional<gl::Shader_Program>& out) {
        auto vsh = FromFileLoadShader<GL_VERTEX_SHADER>(pathVsh, defines);
        auto fsh = FromFileLoadShader<GL_FRAGMENT_SHADER>(pathFsh, defines);
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
            glGenVertexArrays(1, &vao);
            glGenBuffers(1, &vbo);
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

    void push_debug_group(char const* pszFormat, ...) {
        char buffer[256];
        va_list args;
        va_start(args, pszFormat);
        auto len = vsnprintf(buffer, 255, pszFormat, args);
        buffer[255] = 0;
        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, len, buffer);
        va_end(args);
    }

    void pop_debug_group() {
        glPopDebugGroup();
    }

    void draw_ellipsoids(
        gfx::Render_Context_Supplement const& ctx,
        size_t count,
        Vec3 const* centers,
        Vec3 const* sizes,
        Quat const* rotations,
        Vec3 const& color
    ) override {
        if (m_sdf_ellipsoid_batch[0]) {
            // Setup screen quad
            float quad[] = {
                -1,  1,
                 1,  1,
                -1, -1,
                 1, -1,
            };
            GLuint vao, vbo;
            glGenVertexArrays(1, &vao);
            glGenBuffers(1, &vbo);
            glBindVertexArray(vao);

            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferData(GL_ARRAY_BUFFER, 4 * 2 * sizeof(float), quad, GL_STREAM_DRAW);

            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);

            auto const matVP = m_proj * m_view;
            auto const matInvVP = glm::inverse(matVP);

            auto batch_size = SDF_BATCH_SIZE;
            auto order = SDF_BATCH_SIZE_ORDER;
            auto remain = count;
            auto shader = &m_sdf_ellipsoid_batch[order].value();
            glUseProgram(*shader);

            auto locVP = gl::Uniform_Location<Mat4>(*shader, "matVP");
            auto locInvVP = gl::Uniform_Location<Mat4>(*shader, "matInvVP");
            auto locSiz = gl::Uniform_Location<glm::vec4*>(*shader, "vSize");
            auto locTranslation = gl::Uniform_Location<glm::vec4*>(*shader, "vTranslation");
            auto locInvRotation = gl::Uniform_Location<glm::mat4*>(*shader, "matInvRotation");
            auto locSun = gl::Uniform_Location<Vec3>(*shader, "vSun");
            auto locColor = gl::Uniform_Location<Vec3>(*shader, "vColor");

            gl::SetUniformLocation(locVP, matVP);
            gl::SetUniformLocation(locInvVP, matInvVP);

            gl::SetUniformLocation(locSun, ctx.sun ? *ctx.sun : Vec3(10, 10, 10));
            gl::SetUniformLocation(locColor, color);

            glm::vec4 vTranslationArray[SDF_BATCH_SIZE];
            glm::mat4 matInvRotationArray[SDF_BATCH_SIZE];
            glm::vec4 vSizeArray[SDF_BATCH_SIZE];

            for (unsigned off = 0; off < count; off += batch_size) {
                while (remain < batch_size) {
                    batch_size >>= 1;
                    order--;
                    if (remain >= batch_size) {
                        shader = &m_sdf_ellipsoid_batch[order].value();
                        glUseProgram(*shader);

                        locVP = gl::Uniform_Location<Mat4>(*shader, "matVP");
                        locInvVP = gl::Uniform_Location<Mat4>(*shader, "matInvVP");
                        locSiz = gl::Uniform_Location<glm::vec4*>(*shader, "vSize");
                        locTranslation = gl::Uniform_Location<glm::vec4*>(*shader, "vTranslation");
                        locInvRotation = gl::Uniform_Location<glm::mat4*>(*shader, "matInvRotation");
                        locSun = gl::Uniform_Location<Vec3>(*shader, "vSun");
                        locColor = gl::Uniform_Location<Vec3>(*shader, "vColor");

                        gl::SetUniformLocation(locVP, matVP);
                        gl::SetUniformLocation(locInvVP, matInvVP);

                        gl::SetUniformLocation(locSun, ctx.sun ? *ctx.sun : Vec3(10, 10, 10));
                        gl::SetUniformLocation(locColor, color);
                    }
                }

                for (int i = 0; i < batch_size; i++) {
                    auto idx = off + i;
                    vTranslationArray[i] = glm::vec4(centers[idx], 0);
                    matInvRotationArray[i] = Mat3(glm::conjugate(rotations[idx]));
                    vSizeArray[i] = glm::vec4(sizes[idx], 0);
                }

                gl::SetUniformLocationArray(locTranslation, vTranslationArray, batch_size);
                gl::SetUniformLocationArray(locSiz, vSizeArray, batch_size);
                gl::SetUniformLocationArray(locInvRotation, matInvRotationArray, batch_size);

                glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

                remain -= batch_size;
            }

            glDeleteBuffers(1, &vbo);
            glDeleteVertexArrays(1, &vao);
        }
    }

    virtual void new_frame() override {
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

        auto t = 16 - 1000 * flFrameTime;
        if (t > 1) {
            SDL_Delay(t);
        }

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

    std::optional<gl::Shader_Program> m_sdf_ellipsoid_batch[SDF_BATCH_SIZE_ORDER + 1];
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