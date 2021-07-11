// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#include "renderable_manager.h"

namespace topo {
bool
Renderable_Manager::CreateRenderable(
    Renderable_ID *outHandle,
    Model_ID model,
    Material_ID material) {
    if (outHandle == nullptr || model == nullptr || material == nullptr) {
        return false;
    }

    _renderables.push_front({ model, material });
    *outHandle = &_renderables.front();

    return true;
}

void
Renderable_Manager::DestroyRenderable(Renderable_ID id) {
    if (id == nullptr) {
        return;
    }

    std::remove_if(
        _renderables.begin(), _renderables.end(),
        [&](auto const &t) { return &t == id; });
}

void
Renderable_Manager::GetModelAndMaterial(
    Renderable_ID renderable,
    Model_ID *outModel,
    Material_ID *outMaterial) {
    if (renderable != nullptr && outModel != nullptr
        && outMaterial != nullptr) {
        auto r = ((Renderable *)renderable);
        *outModel = r->model;
        *outMaterial = r->material;
        return;
    }

    std::abort();
}
}
