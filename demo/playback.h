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
        IApplication *app)
        : _demo(demo)
        , _currentTime(0)
        , _app(app)
        , _visTree()
        , _visFoliage() { }

    bool
    step(float dt);

    void
    render(topo::IRender_Queue *rq);

    void
    setLeafTexture(topo::Texture_ID tex) {
        _texLeaf = tex;
    }

protected:
    void
    beginRegenerateRenderable();
    void
    regenerateRenderable();
    void
    regenerateRenderableAfter();
    void
    oneshotGrow();
    void
    oneshotGrowAfter();

    bool
    doOneshot(float dt);

    bool
    doTimelapse(float dt);

    static void
    regenerateRenderable(uv_work_t *work);
    static void
    regenerateRenderableAfter(uv_work_t *work, int status);

    static void
    oneshotGrow(uv_work_t *work);
    static void
    oneshotGrowAfter(uv_work_t *work, int status);

private:
    Demo _demo;
    float _currentTime;

    IApplication *_app;

    Trigen_Mesh _mesh, _meshFoliage;
    uv_work_t _workMeshRegen;

    uv_work_t _workGrow;
    bool _generatingVisuals = false;

    topo::Texture_ID _texLeaf = nullptr;

    Visual _visTree;
    Visual _visFoliage;
};
