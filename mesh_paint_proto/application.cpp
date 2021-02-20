// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"
#include "application.h"
#include "events.h"

#include <r_sdl.h>
#include <r_queue.h>
#include <imgui.h>

#include <arcball_camera.h>

/*
 * 
 */
class Arcball_Camera_Event_Handler : public IEvent_Handler {
public:
    Arcball_Camera_Event_Handler(Arcball_Camera *camera) : _camera(camera) {
    }

    bool on_event(SDL_Event const &ev, float delta) override {
        switch (ev.type) {
        case SDL_MOUSEBUTTONDOWN:
            _camera->mouse_down(ev.button.x, ev.button.y);
            return true;
        case SDL_MOUSEBUTTONUP:
            _camera->mouse_up(ev.button.x, ev.button.y);
            return true;
        case SDL_MOUSEMOTION:
            _camera->mouse_move(ev.button.x, ev.button.y);
            return true;
        case SDL_MOUSEWHEEL:
            _camera->mouse_wheel(ev.wheel.y);
            return true;
        }

        return false;
    }

private:
    Arcball_Camera *_camera;
};

class Application : public IApplication {
public:
    Application(
        std::unique_ptr<gfx::ISDL_Window> &&window,
        std::unique_ptr<Arcball_Camera> &&camera) :
    _window(std::move(window)),
    _camera(std::move(camera)),
    _camera_ev_handler(_camera.get()) {

        unsigned r_width, r_height;
        _window->get_resolution(&r_width, &r_height);
        _camera->set_screen_size(r_width, r_height);
    }

    int run() override {
        gfx::Render_Queue rq(512 * 1024);
        SDL_Event ev;
        double delta = 0.01;

        Chain_Of_Responsibility<SDL_Event, IEvent_Handler *> event_handler;
        event_handler.attach(&_camera_ev_handler);

        while (!_quit) {
            while (_window->poll_event(&ev)) {
                if (!event_handler.handle(ev, delta)) {
                    switch (ev.type) {
                    case SDL_QUIT:
                        _quit = true;
                        break;
                    }
                }
            }

            _window->set_camera(_camera->get_view_matrix());
            _window->new_frame();

            if (ImGui::Begin("Window")) {
            }
            ImGui::End();

            rq.execute(_window.get());
            delta = _window->present();
        }

        return 0;
    }

private:
    bool _quit = false;

    std::unique_ptr<gfx::ISDL_Window> _window;
    std::unique_ptr<Arcball_Camera> _camera;
    Arcball_Camera_Event_Handler _camera_ev_handler;
};

std::unique_ptr<IApplication> make_application(
    std::unique_ptr<gfx::ISDL_Window> &&window
) {
    auto camera = create_arcball_camera();
    return std::make_unique<Application>(
        std::move(window),
        std::move(camera)
    );
}
