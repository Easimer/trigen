// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"
#include "application.h"
#include "events.h"
#include "session.h"

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

class Session_Creation_Dialog {
public:
    void draw() {
        if (ImGui::Begin("Open simulation image", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::InputText("Path", _path_buf, 2048);
            ImGui::SameLine();
            if (ImGui::Button("OK")) {
                _ready = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel")) {
                _close = true;
            }
        }
        ImGui::End();
    }

    bool ready() const { return _ready; }
    bool close() const { return _close; }
    char const *path() { return _path_buf; }
private:
    bool _ready = false;
    bool _close = false;
    char _path_buf[2048];
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

            if (ImGui::BeginMainMenuBar()) {
                if (ImGui::BeginMenu("File")) {
                    if (ImGui::MenuItem("Open")) {
                        _session_creation_dialog = Session_Creation_Dialog();
                    }
                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Windows")) {
                    for (auto &session : _sessions) {
                        bool is_current = session.get() == _current_session;
                        if (ImGui::MenuItem(session->title(), nullptr, is_current)) {
                            _current_session = session.get();
                        }
                    }
                    ImGui::EndMenu();
                }
            }
            ImGui::EndMainMenuBar();

            if (_session_creation_dialog) {
                _session_creation_dialog->draw();
                if (_session_creation_dialog->close()) {
                    _session_creation_dialog.reset();
                }
                if (_session_creation_dialog->ready()) {
                    create_session(_session_creation_dialog->path());
                    _session_creation_dialog.reset();
                }
            }

            if (_current_session != nullptr) {
                ImGui::Begin("Session", nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize);
                ImGui::Text("%s", _current_session->title());
                ImGui::Separator();
                ImGui::Text("Marching cubes parameters");
                auto &mc_params = _current_session->marching_cubes_params();
                ImGui::DragInt("Subdivisions", &mc_params.subdivisions);

                ImGui::Separator();
                if (ImGui::Button("Generate mesh")) {
                    _current_session->do_generate_mesh();
                }
                if (ImGui::Button("Paint mesh")) {
                    _current_session->do_paint_mesh();
                }
                ImGui::End();
                _current_session->render(&rq);
            }

            rq.execute(_window.get());
            delta = _window->present();
        }

        return 0;
    }

    void create_session(char const *path) {
        auto session = make_session(path);
        _current_session = session.get();
        _sessions.push_front(std::move(session));
    }

    void remove_session(ISession *session) {
        std::remove_if(_sessions.begin(), _sessions.end(), [&](auto const &s) { return s.get() == session; });

        if (_current_session == session) {
            if (_sessions.size() > 0) {
                _current_session = _sessions.front().get();
            } else {
                _current_session = nullptr;
            }
        }
    }

private:
    bool _quit = false;

    std::unique_ptr<gfx::ISDL_Window> _window;
    std::unique_ptr<Arcball_Camera> _camera;
    Arcball_Camera_Event_Handler _camera_ev_handler;

    std::list<std::unique_ptr<ISession>> _sessions;
    ISession *_current_session = nullptr;
    std::optional<Session_Creation_Dialog> _session_creation_dialog;
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
