// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: editor entry point
//

#include "stdafx.h"
#include <type_traits>
#include "application.h"
#include "r_renderer.h"
#include "r_queue.h"
#include "softbody.h"
#include "softbody_renderer.h"
#include "events.h"
#include <imgui.h>

class IEvent_Handler {
public:
    virtual bool on_event(SDL_Event const& ev, float delta) = 0;

    class Event_Handler_Caller {
    public:
        bool operator()(IEvent_Handler* h, SDL_Event const& ev, float delta) {
            return h->on_event(ev, delta);
        }
    };

    using caller_t = Event_Handler_Caller;
};

struct Renderer_Preferences_Screen : public IEvent_Handler {
public:
    Renderer_Preferences_Screen(gfx::IRenderer* r) : m_renderer(r) {}

    bool on_event(SDL_Event const& ev, float delta) override {
        if (ev.type == SDL_KEYUP && ev.key.keysym.sym == SDLK_TAB) {
            m_shown = !m_shown;
            return true;
        } else {
            return false;
        }
    }

    void draw() {
        if (m_shown) {
            if (ImGui::Begin("Preferences")) {
                ImGui::Text("Resolution");
                draw_res_button(1280, 720);
                draw_res_button(1600, 900);
                draw_res_button(1920, 1080);
            }
            ImGui::End();
        }
    }

private:
    void draw_res_button(unsigned width, unsigned height) {
        char buf[64];
        snprintf(buf, 63, "%ux%u", width, height);
        if (ImGui::Button(buf)) {
            m_renderer->change_resolution(&width, &height);
        }
    }

private:
    bool m_shown = false;
    gfx::IRenderer* m_renderer;
};

struct Arcball_Camera : public IEvent_Handler {
public:
    Arcball_Camera(gfx::IRenderer* r) : m_renderer(r), q_down(1, 0, 0, 0), q_now(1, 0, 0, 0) {}

    // @return Should the caller ignore this event
    bool on_event(SDL_Event const& ev, float delta) override {
        bool ret = false;
        switch (ev.type) {
            case SDL_MOUSEMOTION:
            {
                if (!alt_held) {
                    ret = true;
                    if (ev.motion.state & SDL_BUTTON_LEFT) {
                        step_arcball(ev.motion.x, ev.motion.y);
                    } else {
                    }
                }

                break;
            }
            case SDL_MOUSEBUTTONDOWN:
            {
                if (ev.button.button == SDL_BUTTON_LEFT) {
                    begin_arcball(ev.button.x, ev.button.y);
                    q_down = q_now;
                }
                break;
            }
            case SDL_MOUSEBUTTONUP:
            {
                if (ev.button.button == SDL_BUTTON_LEFT) {
                    end_arcball(ev.button.x, ev.button.y);
                }
                break;
            }
            case SDL_MOUSEWHEEL:
            {
                auto wheel = (ev.wheel.direction == SDL_MOUSEWHEEL_FLIPPED) ? -ev.wheel.y : ev.wheel.y;
                if (wheel > 0) {
                    distance *= 2.0f;
                } else {
                    distance /= 2.0f;
                }

                if (distance < 0.1) distance = 0.1;
                printf("distance: %f\n", distance);
                update_matrix();
                break;
            }
            case SDL_KEYUP:
            {
                switch (ev.key.keysym.sym) {
                case SDLK_LALT:
                    alt_held = false;
                    break;
                }
                break;
            }
            case SDL_KEYDOWN:
            {
                switch (ev.key.keysym.sym) {
                case SDLK_LALT:
                    alt_held = true;
                    break;
                }
            }
        }

        return ret;
    }

protected:
    void begin_arcball(unsigned sx, unsigned sy) {
        unsigned w, h;
        m_renderer->get_resolution(&w, &h);
        sy = h - sy;
        auto c = Vec2(w / 2.0f, h / 2.0f);
        auto r = Vec2(w / 2.0f, h / 2.0f);

        Vec3 temp;
        if (get_arcball_coords(&temp, c, r, sx, sy)) {
            v0 = temp;
        }
    }

    void end_arcball(unsigned sx, unsigned sy) {
        q_down = q_now;
    }

    void step_arcball(unsigned sx, unsigned sy) {
        unsigned w, h;
        m_renderer->get_resolution(&w, &h);
        sy = h - sy;
        auto c = Vec2(w / 2.0f, h / 2.0f);
        auto r = Vec2(w / 2.0f, h / 2.0f);

        Vec3 v1;
        if (get_arcball_coords(&v1, c, r, sx, sy)) {
            auto q_drag = Quat(glm::dot(v0, v1), glm::cross(v0, v1));
            q_now = glm::normalize(q_drag * q_down);
            printf("q_drag %f %f %f %f\n", q_drag.w, q_drag.x, q_drag.y, q_drag.z);
            printf("q_now %f %f %f %f\n", q_now.w, q_now.x, q_now.y, q_now.z);

            update_matrix();
        }
    }

    void update_matrix() {
        auto pos = q_now * Vec3(0, 0, distance) * glm::conjugate(q_now);
        auto mat = glm::lookAt(pos, Vec3(0, 0, 0), Vec3(0, 1, 0));
        m_renderer->set_camera(mat);

        /*
        printf("VIEW:\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n %f %f %f %f\n",
            mat[0][0], mat[0][1], mat[0][2], mat[0][3],
            mat[1][0], mat[1][1], mat[1][2], mat[1][3],
            mat[2][0], mat[2][1], mat[2][2], mat[2][3],
            mat[3][0], mat[3][1], mat[3][2], mat[3][3]
        );
        */
    }

    bool get_arcball_coords(Vec3* out, Vec2 const& c, Vec2 const& r, unsigned sx, unsigned sy) {
        auto s = Vec2(sx, sy);
        auto v = (s - c) / r;
        auto v_len = glm::length(v);
        printf("v %f %f\n", v.x, v.y);
        printf("v_len %f\n", v_len);
        auto z = glm::sqrt(1 - v_len * v_len);
        *out = glm::normalize(Vec3(v.x, v.y, z));
        printf("out %f %f %f\n", out->x, out->y, out->z);
        return !glm::isnan(z);
    }
private:
    gfx::IRenderer* m_renderer;
    float distance = 10.0f;
    bool alt_held = false;

    Vec3 v0;
    Quat q_down, q_now;
};

void app_main_loop() {
    bool quit = false;

    // Renderer setup
    unsigned r_width = 1280, r_height = 720;
    gfx::Renderer_Config render_cfg = {
        r_width, r_height,
    };
    auto renderer = gfx::make_renderer(render_cfg);
    Renderer_Preferences_Screen r_prefs(renderer);

    // Simulation setup
    sb::Config sim_cfg = {
        Vec3(0, 0, 0), // seed_position
    };
    auto sim = sb::create_simulation(sim_cfg);

    // Sun
    float sun_angle = 0.0f;
    double delta = 0.01f;

    // Camera
    Arcball_Camera cam(renderer);

    Chain_Of_Responsibility<SDL_Event, IEvent_Handler*> event_handler;
    event_handler.attach(&r_prefs, &cam);

    while (!quit) {
        SDL_Event ev;
        gfx::Render_Queue rq(512 * 1024);
        while (renderer->pump_event_queue(ev)) {
            if (!event_handler.handle(ev, delta)) {
                switch (ev.type)
                {
                    case SDL_QUIT:
                    {
                        quit = true;
                        break;
                    }
                    case SDL_MOUSEMOTION:
                    {
                        break;
                    }
                    case SDL_KEYUP:
                    {
                        switch (ev.key.keysym.sym) {
                        case SDLK_r:
                            sb::destroy_simulation(sim);
                            sim = sb::create_simulation(sim_cfg);
                            break;
                        }
                        break;
                    }
                    case SDL_KEYDOWN:
                    {
                        switch (ev.key.keysym.sym) {
                        }
                    }
                }
            }
        }
        renderer->new_frame();

        r_prefs.draw();

        sun_angle = glm::mod(sun_angle + delta / 16.0f, glm::pi<double>());
        auto sun_pos = Vec3(1000 * glm::cos(sun_angle), 1000 * glm::sin(sun_angle), 0.0f);
        sb::set_light_source_position(sim, sun_pos);
        for (int i = 0; i < 1; i++) {
            sb::step(sim, delta);
        }
        render_softbody_simulation(&rq, sim);

        rq.execute(renderer);

        if (ImGui::Begin("Sun")) {
            ImGui::Text("Angle:    %f\n", sun_angle);
            ImGui::Text("Position: %f %f %f\n", sun_pos.x, sun_pos.y, sun_pos.z);
        }
        ImGui::End();

        delta = renderer->present();
    }

    sb::destroy_simulation(sim);

    gfx::destroy_renderer(renderer);
}
