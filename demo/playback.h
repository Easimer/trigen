// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#pragma once

#include <vector>
#include "scene_loader.h"
#include <topo.h>

struct Visual {
    topo::IInstance *renderer = nullptr;
    topo::Texture_ID texDiffuse = nullptr;
    topo::Texture_ID texNormal = nullptr;
    topo::Material_ID material = nullptr;
    topo::Model_ID model = nullptr;
    topo::Renderable_ID renderable = nullptr;

    Visual &
        operator=(Visual&& other) noexcept {
        clear();
        std::swap(renderer, other.renderer);
        std::swap(texDiffuse, other.texDiffuse);
        std::swap(texNormal, other.texNormal);
        std::swap(material, other.material);
        std::swap(model, other.model);
        std::swap(renderable, other.renderable);
        return *this;
    }

    ~Visual() { clear(); }

    void
    clear() {
        if (*this) {
            renderer->DestroyRenderable(renderable);
            renderer->DestroyMaterial(material);
            renderer->DestroyModel(model);
            renderer->DestroyTexture(texDiffuse);
            renderer->DestroyTexture(texNormal);
            renderer = nullptr;
            texDiffuse = nullptr;
            texNormal = nullptr;
            material = nullptr;
            model = nullptr;
            renderable = nullptr;
        }
    }

    constexpr operator bool() const noexcept { return renderable != nullptr; }
};

class Playback {
public:
    Playback(
        Demo const &demo,
        Trigen_Session simulation,
        topo::IInstance *renderer)
        : _demo(demo)
        , _currentTime(0)
        , _simulation(simulation)
        , _renderer(renderer) { }

    bool
    step(float dt);

    void
    render(topo::IRender_Queue *rq);

    protected:
    void
    regenerateRenderable();

    bool
    doOneshot(float dt);

    bool
    doTimelapse(float dt);

private:
    Demo _demo;
    float _currentTime;

    Trigen_Session _simulation;
    topo::IInstance *_renderer;

    Visual _visTree;
    Visual _visFoliage;
};
