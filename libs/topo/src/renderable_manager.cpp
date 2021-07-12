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

    _models.push_front({ model, material });
    auto *renderableModel = &_models.front();

    Renderable r;
    r.kind = RENDERABLE_MODEL;
    r.model = renderableModel;

    _renderables.push_front(r);
    *outHandle = &_renderables.front();

    return true;
}

void
Renderable_Manager::DestroyRenderable(Renderable_ID id) {
    if (id == nullptr) {
        return;
    }

    auto *renderable = (Renderable *)id;

    switch (renderable->kind) {
    case RENDERABLE_MODEL:
        std::remove_if(
            _models.begin(), _models.end(),
            [&](auto const &t) { return &t == renderable->model; });
        break;
    case RENDERABLE_LINES:
        std::remove_if(
            _lines.begin(), _lines.end(),
            [&](auto const &t) { return &t == renderable->lines; });
        break;
    }

    std::remove_if(
        _renderables.begin(), _renderables.end(),
        [&](auto const &t) { return &t == id; });
}

Renderable_Manager::Renderable_Kind
Renderable_Manager::GetRenderableKind(Renderable_ID id) {
    if (id == nullptr) {
        std::abort();
    }

    auto *renderable = (Renderable *)id;

    return renderable->kind;
}

void
Renderable_Manager::GetModelAndMaterial(
    Renderable_ID id,
    Model_ID *outModel,
    Material_ID *outMaterial) {
    if (id == nullptr || outModel == nullptr
        || outMaterial == nullptr) {
        std::abort();
    }

    auto *renderable = (Renderable *)id;

    if (renderable->kind != RENDERABLE_MODEL) {
        std::abort();
    }

    *outModel = renderable->model->model;
    *outMaterial = renderable->model->material;
}

bool
Renderable_Manager::CreateRenderableLinesStreaming(
    Renderable_ID *outHandle,
    glm::vec3 const *endpoints,
    size_t lineCount,
    glm::vec3 const &colorBegin,
    glm::vec3 const &colorEnd) {
    if (outHandle == nullptr || endpoints == nullptr || lineCount == 0) {
        return false;
    }

    std::vector<glm::vec3> endpointsVec;
    endpointsVec.reserve(2 * lineCount);

    for (size_t i = 0; i < lineCount; i++) {
        endpointsVec.push_back(endpoints[i * 2 + 0]);
        endpointsVec.push_back(endpoints[i * 2 + 1]);
    }

    _lines.push_front({ std::move(endpointsVec), colorBegin, colorEnd });
    auto *renderableLines = &_lines.front();

    Renderable r;
    r.kind = RENDERABLE_LINES;
    r.lines = renderableLines;

    _renderables.push_front(r);
    *outHandle = &_renderables.front();

    return true;
}

void
Renderable_Manager::GetLines(
    Renderable_ID id,
    glm::vec3 const **endpoints,
    size_t *lineCount,
    glm::vec3 *color) {
    if (id == nullptr || endpoints == nullptr
        || lineCount == nullptr || color == nullptr) {
        std::abort();
    }

    auto *renderable = (Renderable *)id;

    if (renderable->kind != RENDERABLE_LINES) {
        std::abort();
    }

    *endpoints = renderable->lines->endpoints.data();
    *lineCount = renderable->lines->endpoints.size() / 2;
    *color = renderable->lines->colorBegin;
}

}
