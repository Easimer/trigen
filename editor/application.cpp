// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: editor entry point
//

#include "stdafx.h"
#include "application.h"
#include "r_renderer.h"
#include "r_queue.h"
#include "softbody.h"
#include "softbody_renderer.h"
#include <imgui.h>

void app_main_loop() {
    bool quit = false;
    auto renderer = gfx::make_renderer();
    sb::Config sim_cfg;
    sim_cfg.seed_position = Vec3(0, 0, 0);
    auto sim = sb::create_simulation(sim_cfg);
    float sun_angle = 0.0f;
    double delta = 0.01f;

    while (!quit) {
        SDL_Event ev;
        gfx::Render_Queue rq(512 * 1024);
        while (renderer->pump_event_queue(ev)) {
            switch (ev.type) {
            case SDL_QUIT:
                quit = true;
                break;
            case SDL_MOUSEMOTION:
            {
                auto pos = Vec3(ev.motion.x - 1280 / 2.0f, -(ev.motion.y - 720 / 2.0f), 0);
                sun_angle = glm::atan<float>(pos.y, pos.x);
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
            }
        }
        renderer->new_frame();

        // sun_angle += glm::two_pi<float>() / 4 * delta;
        auto sun_pos = Vec3(1000 * glm::cos(sun_angle), 1000 * glm::sin(sun_angle), 0.0f);
        printf("sun: %f %f\n", sun_pos.x, sun_pos.y);
        sb::set_light_source_position(sim, sun_pos);
        sb::step(sim, delta);
        render_softbody_simulation(&rq, sim);

        rq.execute(renderer);
        delta = renderer->present();
    }

    sb::destroy_simulation(sim);

    gfx::destroy_renderer(renderer);
}
