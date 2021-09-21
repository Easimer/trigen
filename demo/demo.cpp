// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#include <topo.h>
#include <topo_sdl.h>

#include <arcball_camera.h>
#include <trigen.h>

#include "scene.h"

int
main(int argc, char **argv) {
    topo::Surface_Config surf;
    surf.width = 1024;
    surf.height = 1024;
    surf.title = "Demo";
    auto wnd = topo::MakeWindow(surf);

    bool shutdown = false;
    SDL_Event ev;

    Trigen_Session simulation = nullptr;
    Trigen_Parameters params;
    params.flags = Trigen_F_PreferCPU;
    params.particle_count_limit = 512;
    params.seed_position[0] = 0;
    params.seed_position[1] = 0;
    params.seed_position[2] = 0;
    params.density = 1;
    params.phototropism_response_strength = 1;
    params.aging_rate = 0.1;
    params.branching_probability = 0.25;
    params.branch_angle_variance = 3.141592;
    params.surface_adaption_strength = 1.0;
    params.attachment_strength = 1.0;
    params.stiffness = 0.2;
    auto rc = Trigen_CreateSession(&simulation, &params);
    if (rc != Trigen_OK) {
        fprintf(stderr, "CreateSession has failed, rc=%d\n", rc);
        return 1;
    }

    wnd->BeginModelManagement();
    auto scene = MakeScene(Scene::K_BASIC_CUBE, wnd.get(), simulation);
    wnd->FinishModelManagement();

    auto camera = create_arcball_camera();
    camera->set_screen_size(surf.width, surf.height);

    float ang = 0;

    while (!shutdown) {
        wnd->NewFrame();
        while (wnd->PollEvent(&ev)) {
            switch (ev.type) {
            case SDL_QUIT: {
                shutdown = true;
                break;
            }
            case SDL_WINDOWEVENT: {
                switch (ev.window.event) {
                case SDL_WINDOWEVENT_RESIZED: {
                    camera->set_screen_size(ev.window.data1, ev.window.data2);
                    break;
                }
                }
            }
            case SDL_MOUSEMOTION: {
                camera->mouse_move(ev.motion.x, ev.motion.y);
                break;
            }
            case SDL_MOUSEBUTTONDOWN: {
                camera->mouse_down(ev.button.x, ev.button.y);
                break;
            }
            case SDL_MOUSEBUTTONUP: {
                camera->mouse_up(ev.button.x, ev.button.y);
                break;
            }
            case SDL_MOUSEWHEEL: {
                camera->mouse_wheel(ev.wheel.y);
                break;
            }
            }
        }

        wnd->SetEyeViewMatrix(camera->get_view_matrix());
        auto *rq = wnd->BeginRendering();

        scene->Render(rq);

        ang += 1 / 60.0f;
        auto pos = glm::vec3(10 * glm::cos(ang), 0, 10 * glm::sin(ang));
        rq->AddLight({ 1, 1, 1, 1 }, { pos, glm::quat(), glm::vec3(1, 1, 1) }, true);

        wnd->FinishRendering();
        wnd->Present();
    }

    wnd->BeginModelManagement();
    scene->Cleanup(wnd.get());
    wnd->FinishModelManagement();
    Trigen_DestroySession(simulation);
    return 0;
}