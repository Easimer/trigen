// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#pragma once

#include <list>

#include <topo.h>

namespace topo {
class Renderable_Manager {
public:
    enum Renderable_Kind {
        RENDERABLE_MODEL,
        RENDERABLE_LINES,
    };

    bool
    CreateRenderable(
        Renderable_ID *outHandle,
        Model_ID model,
        Material_ID material);

    void
    DestroyRenderable(Renderable_ID renderable);

    Renderable_Kind
    GetRenderableKind(Renderable_ID renderable);

    void
    GetModelAndMaterial(
        Renderable_ID renderable,
        Model_ID *outModel,
        Material_ID *outMaterial);

    bool
    CreateRenderableLinesStreaming(
        Renderable_ID *outHandle,
        glm::vec3 const *endpoints,
        size_t lineCount,
        glm::vec3 const &colorBegin,
        glm::vec3 const &colorEnd);

    void
    GetLines(
        Renderable_ID renderable,
        glm::vec3 const **endpoints,
        size_t *lineCount,
        glm::vec3 *color);

private:
    struct Renderable_Model {
        Model_ID model;
        Material_ID material;
    };

    struct Renderable_Lines {
        std::vector<glm::vec3> endpoints;
        glm::vec3 colorBegin, colorEnd;
    };

    struct Renderable {
        Renderable_Kind kind;

        union {
            Renderable_Model *model;
            Renderable_Lines *lines;
        };
    };

    std::list<Renderable> _renderables;

    std::list<Renderable_Model> _models;
    std::list<Renderable_Lines> _lines;
};
}
