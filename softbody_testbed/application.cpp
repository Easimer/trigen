// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: main loop
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
#include <raymarching.h>
#include "arcball_camera.h"

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

    bool draw() {
        bool ret = false;
        if (m_shown) {
            if (ImGui::Begin("Preferences")) {
                ImGui::Text("Resolution");
                ret |= draw_res_button(1280, 720);
                ret |= draw_res_button(1600, 900);
                ret |= draw_res_button(1920, 1080);
            }
            ImGui::End();
        }
        return ret;
    }

private:
    bool draw_res_button(unsigned width, unsigned height) {
        char buf[64];
        snprintf(buf, 63, "%ux%u", width, height);
        if (ImGui::Button(buf)) {
            m_renderer->change_resolution(&width, &height);
            return true;
        }
        return false;
    }

private:
    bool m_shown = false;
    gfx::IRenderer* m_renderer;
};

static unsigned g_sim_speed = 1;

static void edit_unsigned(char const* l, unsigned* v) {
    int t = (int)*v;
    ImGui::InputInt(l, &t);
    if (t < 0) t = 0;
    *v = (unsigned)t;
}

static bool display_simulation_config(sb::Config& cfg) {
    ImGui::InputFloat3("Seed position", &cfg.seed_position.x);
    ImGui::InputFloat("Density", &cfg.density);
    ImGui::InputFloat("Attachment str.", &cfg.attachment_strength);
    ImGui::InputFloat("Surface adaption str.", &cfg.surface_adaption_strength);
    ImGui::InputFloat("Initial stiffness", &cfg.stiffness);
    ImGui::InputFloat("Aging rate", &cfg.aging_rate);
    ImGui::InputFloat("Phototropism response str.", &cfg.phototropism_response_strength);
    ImGui::InputFloat("Branching probability", &cfg.branching_probability);
    ImGui::InputFloat("Branch angle variance", &cfg.branch_angle_variance);

    edit_unsigned("Particle count limit", &cfg.particle_count_limit);
    edit_unsigned("Simulation speed", &g_sim_speed);

    return ImGui::Button("Reset simulation");
}

struct Falling_Edge {
public:
    void operator=(bool input) {
        if (v && !input) {
            s = true;
        }
        v = input;
    }

    operator bool() {
        return get();
    }

    bool get() {
        auto ret = s;
        s = false;
        return ret;
    }
private:
    bool v = false, s = false;
};

void app_main_loop() {
    bool quit = false;

    // Renderer setup
    unsigned r_width = 1280, r_height = 720;
    gfx::Renderer_Config render_cfg = {
        r_width, r_height,
        "softbody testbed",
    };
    auto renderer = gfx::make_renderer(render_cfg);
    Renderer_Preferences_Screen r_prefs(renderer);

    // Simulation setup
    sb::Config sim_cfg = {
        sb::Extension::Debug_Cloth,
        Vec3(0, 0, 0), // seed_position
        1.0f, // density
        1.0f, // attachment_strength
        1.0f, // surface_adaption_strength
        0.2f, // stiffness
        0.1f, // aging_rate
        1.0f, // phototropism_response_strength
        0.25f, // branching_probability
        glm::pi<float>(), // branch_angle_variance
        128, // particle_count_limit
    };
    sb::Unique_Ptr<sb::ISoftbody_Simulation> sim;
    bool bDoTick = false;
    Falling_Edge bStep;
    sb::Unique_Ptr<sb::ISingle_Step_State> pSingleStep;
    char stateDescription[128] = { '\0' };
    Softbody_Render_Parameters render_params = {};

    auto sdf_box_100_100 = std::bind(&sdf::box, glm::vec3(100, 2, 100), std::placeholders::_1);
    auto sdf_wall_100_100 = std::bind(&sdf::box, glm::vec3(2, 100, 100), std::placeholders::_1);
    auto sdf_box = sdf::translate(sdf_box_100_100, glm::vec3(0, -1, 0));
    auto sdf_wall = sdf::translate(sdf_wall_100_100, glm::vec3(3, 0, 0));
    auto sdf_sphere_3 = std::bind(&sdf::sphere, 3, std::placeholders::_1);
    auto sdf_sphere = sdf::translate(sdf_sphere_3, glm::vec3(0, 6, 0));

    auto reset_simulation = [&]() {
        sim.reset();
        sim = sb::create_simulation(sim_cfg);
        sim->add_collider(sdf_box);
        // sim->add_collider(sdf_wall);
        sim->add_collider(sdf_sphere);
    };

    reset_simulation();

    // Sun
    float sun_angle = 0.0f;
    double delta = 0.01f;
    float flDeltaDivider = 1;

    // Camera
    Arcball_Camera* cam = create_arcball_camera();
    cam->set_screen_size(r_width, r_height);

    Chain_Of_Responsibility<SDL_Event, IEvent_Handler*> event_handler;
    event_handler.attach(&r_prefs, cam);

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
                            sim.reset();
                            sim = sb::create_simulation(sim_cfg);
                            break;
                        }
                        break;
                    }
                }
            }
        }

        renderer->set_camera(cam->get_view_matrix());
        renderer->new_frame();

        if (r_prefs.draw()) {
            // renderer preferences changed, query screen size
            renderer->get_resolution(&r_width, &r_height);
            cam->set_screen_size(r_width, r_height);
        }

        auto steps = (g_sim_speed > 0) ? g_sim_speed : 1;
        delta /= flDeltaDivider;

        sun_angle = glm::mod(sun_angle + steps * delta / 16.0f, glm::pi<double>());
        auto sun_pos = Vec3(1000 * glm::cos(sun_angle), 1000 * glm::sin(sun_angle), 0.0f);
        sim->set_light_source_position(sun_pos);

        if (ImGui::Begin("Controls")) {
            ImGui::Checkbox("Tick", &bDoTick);
            ImGui::InputFloat("Time step divider", &flDeltaDivider);
            bStep = ImGui::Button("Step");
            ImGui::Checkbox("Draw positions", &render_params.draw_positions);
            ImGui::Checkbox("Draw centers of masses", &render_params.draw_center_of_mass);
            ImGui::Checkbox("Draw goal positions", &render_params.draw_goal_position);
            ImGui::Checkbox("Draw bind pose", &render_params.draw_bind_pose);
            ImGui::Text(stateDescription);

            if (flDeltaDivider < 1) flDeltaDivider = 1;
        }
        ImGui::End();

        if (bStep) {
            if (pSingleStep == NULL) {
                pSingleStep = sim->begin_single_step();
            }

            pSingleStep->step();
            pSingleStep->get_state_description(128, stateDescription);
        }

        if (bDoTick) {
            if (pSingleStep != NULL) {
                pSingleStep.reset();
                snprintf(stateDescription, 127, "(not single stepping)");
            }

            for (unsigned i = 0; i < steps; i++) {
                sim->step(delta);
            }
        }

        render_params.sun_position = sun_pos;
        render_softbody_simulation(&rq, sim.get(), render_params);

        rq.execute(renderer);

        if (ImGui::Begin("Sun")) {
            ImGui::Text("Angle:    %f\n", sun_angle);
            ImGui::Text("Position: %f %f %f\n", sun_pos.x, sun_pos.y, sun_pos.z);
        }
        ImGui::End();


        if (ImGui::Begin("Configuration")) {
            if (display_simulation_config(sim_cfg)) {
                reset_simulation();
            }
        }
        ImGui::End();

        delta = renderer->present();
    }

    cam->release();

    gfx::destroy_renderer(renderer);
}
