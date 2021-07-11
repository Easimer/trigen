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
        RENDERABLE_LIGHT,
    };

    bool
    CreateRenderable(
        Renderable_ID *outHandle,
        Model_ID model,
        Material_ID material);

    void
    DestroyRenderable(Renderable_ID renderable);

    void
    GetModelAndMaterial(
        Renderable_ID renderable,
        Model_ID *outModel,
        Material_ID *outMaterial);

private:
    struct Renderable {
        Model_ID model;
        Material_ID material;
    };

    std::list<Renderable> _renderables;
};
}
