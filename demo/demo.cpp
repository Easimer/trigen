// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#include <topo.h>
#include <topo_sdl.h>

#include <arcball_camera.h>
#include <trigen.h>

#include "scene.h"
#include "scene_loader.h"

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

    wnd->BeginModelManagement();
    Scene scene;
    std::vector<Scene::Collider> colliders;
    LoadSceneFromFile("scene_rockywall.json", scene, wnd.get(), &simulation, colliders);
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

        scene.Render(rq);

        ang += 1 / 60.0f;
        auto pos = glm::vec3(30 * glm::cos(ang), 0, 30 * glm::sin(ang));
        rq->AddLight({ 1, 1, 1, 5 }, { pos, glm::quat(), glm::vec3(1, 1, 1) }, true);

        wnd->FinishRendering();
        wnd->Present();
    }

    wnd->BeginModelManagement();
    scene.Cleanup(wnd.get());
    wnd->FinishModelManagement();
    Trigen_DestroySession(simulation);
    return 0;
}