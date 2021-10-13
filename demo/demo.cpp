// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#include <optional>

#include <topo.h>
#include <topo_sdl.h>

#include <arcball_camera.h>
#include <trigen.h>

#include "iapplication.h"
#include "async_image_loader.h"
#include "playback.h"
#include "scene.h"
#include "scene_loader.h"

#include <uv.h>

class Application : public IApplication {
public:
    ~Application() override = default;
    Application()
        : _imageLoader(&_loopMain)
        , _simulation(nullptr) {
        int rc;
        rc = uv_loop_init(&_loopMain);

        topo::Surface_Config surf;
        surf.width = 1024;
        surf.height = 1024;
        surf.title = "Demo";
        _renderer = topo::MakeWindow(surf);

        bool shutdown = false;
        SDL_Event ev;

        _renderer->BeginModelManagement();
        std::vector<Scene::Collider> colliders;
        LoadSceneFromFile(
            "scene_rockywall.json", _scene, this, colliders, _demo);
        _renderer->FinishModelManagement();

        _camera = create_arcball_camera();
        _camera->set_screen_size(surf.width, surf.height);

        _playback.emplace(_demo, this);

        _timeThen = uv_hrtime();

        uv_timer_init(&_loopMain, &_timerRender);
        _timerRender.data = this;

        uv_check_init(&_loopMain, &_checkEvents);
        _checkEvents.data = this;

        uv_timer_init(&_loopMain, &_timerPlayback);
        _timerPlayback.data = this;

        uv_timer_start(&_timerRender, &Application::Render, 0, 16);
        uv_check_start(&_checkEvents, &Application::PumpEvents);
    }

    int
    RunLoop() {
        return uv_run(&_loopMain, UV_RUN_DEFAULT);
    }

    void
    Shutdown() {
        uv_timer_stop(&_timerRender);
        uv_check_stop(&_checkEvents);
        uv_timer_stop(&_timerPlayback);
        uv_close((uv_handle_t*)&_timerRender, nullptr);
        uv_loop_close(&_loopMain);
        _renderer->BeginModelManagement();
        _scene.Cleanup(_renderer.get());
        _renderer->FinishModelManagement();
        Trigen_DestroySession(_simulation);
    }

    void
    PumpEvents() {
        SDL_Event ev;
        while (_renderer->PollEvent(&ev)) {
            switch (ev.type) {
            case SDL_QUIT: {
                _shutdown = true;
                break;
            }
            case SDL_WINDOWEVENT: {
                switch (ev.window.event) {
                case SDL_WINDOWEVENT_RESIZED: {
                    _camera->set_screen_size(ev.window.data1, ev.window.data2);
                    break;
                }
                }
            }
            case SDL_MOUSEMOTION: {
                _camera->mouse_move(ev.motion.x, ev.motion.y);
                break;
            }
            case SDL_MOUSEBUTTONDOWN: {
                _camera->mouse_down(ev.button.x, ev.button.y);
                break;
            }
            case SDL_MOUSEBUTTONUP: {
                _camera->mouse_up(ev.button.x, ev.button.y);
                break;
            }
            case SDL_MOUSEWHEEL: {
                _camera->mouse_wheel(ev.wheel.y);
                break;
            }
            }
        }

    }

    void
    Render() {
        _renderer->NewFrame();

        _renderer->SetEyeViewMatrix(_camera->get_view_matrix());
        auto *rq = _renderer->BeginRendering();

        _scene.Render(rq);
        if (_playback && _renderPlayback) {
            _playback->render(rq);
        }

        _ang += 1 / 60.0f;
        auto pos = glm::vec3(30 * glm::cos(_ang), 0, 30 * glm::sin(_ang));
        rq->AddLight({ 1, 1, 1, 5 }, { pos, glm::quat(), glm::vec3(1, 1, 1) }, true);

        _renderer->FinishRendering();
        _renderer->Present();

        uv_update_time(&_loopMain);

        if (_shutdown) {
            uv_stop(&_loopMain);
        }
    }

    void
    StepPlayback(float dt) {
        _shutdown |= _playback->step(dt);
    }

    IAsync_Image_Loader *
    ImageLoader() override {
        return &_imageLoader;
    }

    topo::IInstance *
    Renderer() override {
        return _renderer.get();
    }

    virtual Trigen_Session
    Simulation() override {
        return _simulation;
    }

    void
    SetSimulation(Trigen_Session sim) override {
        assert(_simulation == nullptr);
        _simulation = sim;
    }

    void
    OnInputTextureLoaded() override {
        if (_inputTexturesRemain > 0) {
            _inputTexturesRemain--;
        }

        if (_inputTexturesRemain == 0) {
            // Begin playback
            printf("[Application] input textures loaded, resuming playback\n");
            uv_timer_start(&_timerPlayback, &Application::StepPlayback, 0, 16);
        }
    }

    void
    OnLeafTextureLoaded(topo::Texture_ID texture) override {
        _texLeaf = texture;
        _playback->setLeafTexture(texture);
    }

    void
    OnTreeVisualsReady() override {
        _renderPlayback = true;
    }

    uv_loop_t *
    Loop() override {
        return &_loopMain;
    }

private:
    static void
    Render(uv_timer_t *idle) {
        auto *app = (Application *)idle->data;
        app->Render();
    }

    static void
    PumpEvents(uv_check_t *check) {
        auto *app = (Application *)check->data;
        app->PumpEvents();
    }

    static void
    StepPlayback(uv_timer_t* timer) {
        auto *app = (Application *)timer->data;
        auto now = uv_hrtime();
        auto dtNano = now - app->_timeThen;
        auto dt = (float)((double)dtNano / 1000000000.0);
        app->_timeThen = now;

        app->StepPlayback(dt);
    }

private:
    topo::UPtr<topo::ISDL_Window> _renderer;
    Trigen_Session _simulation;
    Scene _scene;
    Demo _demo;
    std::optional<Playback> _playback;
    std::unique_ptr<Arcball_Camera> _camera;

    topo::Texture_ID _texLeaf = nullptr;
    float _ang = 0;
    bool _shutdown = false;
    bool _renderPlayback = false;
    unsigned _inputTexturesRemain = 3;

    uv_loop_t _loopMain;
    uv_timer_t _timerRender;
    uv_check_t _checkEvents;
    uv_timer_t _timerPlayback;

    uint64_t _timeThen;

    Async_Image_Loader _imageLoader;
};

int
main(int argc, char **argv) {
    Application app;
    app.RunLoop();
    app.Shutdown();
    return 0;
}