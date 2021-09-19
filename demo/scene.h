// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <memory>
#include <vector>
#include <topo.h>

class Scene {
public:
    virtual ~Scene() = default;

    virtual void
    Cleanup(topo::IInstance *renderer)
        = 0;

    virtual void
    Render(topo::IRender_Queue *rq)
        = 0;

    virtual std::vector<topo::Model_ID>
    LoadObjMesh(topo::IInstance *renderer, char const *path);

    enum Kind {
        K_BASIC_CUBE,
    };
};

std::unique_ptr<Scene>
MakeScene(Scene::Kind kind, topo::IInstance *renderer);
