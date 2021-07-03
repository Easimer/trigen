// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: Renderer with the surface provided by SDL
//

#include <cstdio>
#include "sdl_helper.h"
#include "r_sdl.h"
#include <imgui_impl_opengl3.h>
#include <imgui_impl_sdl.h>

#define GLRES_GLM
#include <glm/gtc/type_ptr.hpp>
#include "glres.h"
#include <optional>
#include <fstream>
#include <sstream>
#include <queue>

#include <Tracy.hpp>

class SDL_Renderer : public sdl::Renderer, public gfx::ISDL_Window {
public:
    SDL_Renderer(gfx::Surface_Config const& cfg, gfx::Renderer_Backend backend) :
        m_width(cfg.width), m_height(cfg.height),
        Renderer(cfg.title, cfg.width, cfg.height,
            SDL_WINDOW_SHOWN | (backend == gfx::Renderer_Backend::OpenGL ? SDL_WINDOW_OPENGL : 0)) {
        if (window != NULL && renderer != NULL) {
            switch (backend) {
            case gfx::Renderer_Backend::OpenGL:
            {
                make_gl_backend();
                break;
            }
            }
            m_uiTimeStart = SDL_GetPerformanceCounter();

            // Notify backend about the resolution
            unsigned w, h;
            w = cfg.width;
            h = cfg.height;
            this->backend->change_resolution(&w, &h);
        }
    }

    void make_gl_backend() {
        glctx = SDL_GL_CreateContext(window);

        SDL_GL_SetSwapInterval(0);
        gladLoadGLLoader(SDL_GL_GetProcAddress);

        IMGUI_CHECKVERSION();
        m_pImGuiCtx = ImGui::CreateContext();
        ImGui::GetIO();
        ImGui::StyleColorsDark();
        ImGui_ImplSDL2_InitForOpenGL(window, glctx);
        this->backend = gfx::make_opengl_renderer(glctx, SDL_GL_GetProcAddress, m_pImGuiCtx);
    }

    ~SDL_Renderer() {
        if (glctx != NULL) {
            backend.reset();
            ImGui_ImplSDL2_Shutdown();
            SDL_GL_DeleteContext(glctx);
        }
    }

    void new_frame() override {
        ZoneScoped;
        backend->new_frame();

        ImGui_ImplSDL2_NewFrame(window);
        ImGui::NewFrame();
    }

    double present() override {
        ZoneScoped;
        ImGui::Render();
        backend->present();

        auto const uiTimeEnd = SDL_GetPerformanceCounter();
        auto const flFrameTime = (uiTimeEnd - m_uiTimeStart) / (double)SDL_GetPerformanceFrequency();

        SDL_GL_SwapWindow(window);

        m_uiTimeStart = SDL_GetPerformanceCounter();

        auto t = 16.6666666 - 1000 * flFrameTime;
        if (t > 1) {
            SDL_Delay(t);
        }

        return flFrameTime;
    }

    void set_camera(glm::mat4 const& view_matrix) override {
        backend->set_camera(view_matrix);
    }

    void get_camera(glm::mat4 &view_matrix, glm::mat4 &projection_matrix) override {
        backend->get_camera(view_matrix, projection_matrix);
    }

    void draw_points(size_t nCount, glm::vec3 const* pPoints, glm::vec3 const& vWorldPosition) override {
        backend->draw_points(nCount, pPoints, vWorldPosition);
    }

    void draw_lines(glm::vec3 const* pEndpoints, size_t nLineCount, glm::vec3 const& vWorldPosition, glm::vec3 const& vStartColor, glm::vec3 const& vEndColor) override {
        backend->draw_lines(pEndpoints, nLineCount, vWorldPosition, vStartColor, vEndColor);
    }

    void change_resolution(unsigned* inout_width, unsigned* inout_height) override {
        ZoneScoped;
        assert(inout_width != NULL);
        assert(inout_height != NULL);

        m_width = *inout_width;
        m_height = *inout_height;
        SDL_SetWindowSize(window, m_width, m_height);

        backend->change_resolution(inout_width, inout_height);
    }

    void get_resolution(unsigned* out_width, unsigned* out_height) override {
        *out_width = m_width;
        *out_height = m_height;
    }

    bool poll_event(SDL_Event* ev) override {
        ZoneScoped;
        auto& io = ImGui::GetIO();
        bool filtered = false;

        do {
            filtered = false;
            if (SDL_PollEvent(ev)) {
                ImGui_ImplSDL2_ProcessEvent(ev);

                if (io.WantCaptureKeyboard) {
                    switch (ev->type) {
                    case SDL_KEYDOWN:
                    case SDL_KEYUP:
                        filtered = true;
                        break;
                    default:
                        break;
                    }
                }
                if (io.WantCaptureMouse) {
                    switch (ev->type) {
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

    void draw_triangle_elements(size_t vertex_count, std::array<float, 3> const* vertices, size_t element_count, unsigned const* elements, glm::vec3 const& vWorldPosition) override {
        backend->draw_triangle_elements(vertex_count, vertices, element_count, elements, vWorldPosition);
    }

    void draw_triangle_elements_with_vertex_color(size_t vertex_count, std::array<float, 3> const* vertices, glm::u8vec3 const* vertex_colors, size_t element_count, unsigned const* elements, glm::vec3 const& vWorldPosition) override {
        backend->draw_triangle_elements_with_vertex_color(vertex_count, vertices, vertex_colors, element_count, elements, vWorldPosition);
    }

    bool upload_texture(gfx::Texture_ID *out_id, unsigned width, unsigned height, gfx::Texture_Format format, void const *image) override {
        return backend->upload_texture(out_id, width, height, format, image);
    }

    void destroy_texture(gfx::Texture_ID id) override {
        return backend->destroy_texture(id);
    }

    bool create_model(gfx::Model_ID *out_id, gfx::Model_Descriptor const *model) override {
        return backend->create_model(out_id, model);
    }

    void destroy_model(gfx::Model_ID id) override {
        return backend->destroy_model(id);
    }

    void draw_textured_triangle_elements(
        gfx::Model_ID model,
        gfx::Material_Unlit const &material,
        gfx::Transform const &transform
    ) override {
        return backend->draw_textured_triangle_elements(model, material, transform);
    }

    void draw_triangle_elements(
        gfx::Model_ID model_handle,
        gfx::Transform const &transform
    ) override {
        return backend->draw_triangle_elements(model_handle, transform);
    }

    void draw_triangle_elements(
        gfx::Render_Parameters const &params,
        gfx::Model_ID model_handle,
        gfx::Transform const &transform
    ) override {
        return backend->draw_triangle_elements(params, model_handle, transform);
    }

    void draw_textured_triangle_elements(
        gfx::Model_ID model,
        gfx::Material_Lit const &material,
        gfx::Transform const &transform) override {
        return backend->draw_textured_triangle_elements(model, material, transform);
    }

    void set_sun_position(glm::vec3 const &position) override {
        backend->set_sun_position(position);
    }

    void
    create_framebuffer(gfx::Framebuffer_ID *out_id, float resolution_scale)
        override {
        backend->create_framebuffer(out_id, resolution_scale);
    }

    void
    destroy_framebuffer(gfx::Framebuffer_ID id) override {
        backend->destroy_framebuffer(id);
    }

    void
    activate_framebuffer(gfx::Framebuffer_ID id) override {
        backend->activate_framebuffer(id);
    }

    void
    draw_framebuffer(gfx::Framebuffer_ID id) override {
        backend->draw_framebuffer(id);
    }

    void
    clear(glm::vec4 color) override {
        backend->clear(color);
    }

    void
    draw_transparent_model(
        gfx::Model_ID model,
        gfx::Texture_ID diffuse,
        gfx::Transform const &transform) override {
        backend->draw_transparent_model(model, diffuse, transform);
    }

private:
    std::unique_ptr<gfx::IRenderer> backend;
    int m_width, m_height;
    void* glctx;
    ImGuiContext* m_pImGuiCtx;
    gfx::Renderer_Backend backend_kind;
    decltype(SDL_GetPerformanceCounter()) m_uiTimeStart;
};


std::unique_ptr<gfx::ISDL_Window> gfx::make_window(gfx::Surface_Config const& cfg, gfx::Renderer_Backend backend) {
    SDL_Init(SDL_INIT_EVERYTHING);
    return std::make_unique<SDL_Renderer>(cfg, backend);
}
