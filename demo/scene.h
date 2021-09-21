// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <memory>
#include <vector>
#include <topo.h>
#include <trigen.h>

class Scene {
public:
    struct Collider {
        Collider(
            topo::Model_ID hVisual,
            Trigen_Collider hSimulation)
            : hVisual(hVisual)
            , hSimulation(hSimulation) { }
        topo::Model_ID hVisual;
        Trigen_Collider hSimulation;
    };

    virtual ~Scene() = default;

    virtual void
    Cleanup(topo::IInstance *renderer)
        = 0;

    virtual void
    Render(topo::IRender_Queue *rq)
        = 0;

    virtual std::vector<Collider>
    LoadObjMeshCollider(
        topo::IInstance *renderer,
        Trigen_Session simulation,
        Trigen_Transform const &transform,
        char const *path);

    enum Kind {
        K_BASIC_CUBE,
    };

};

std::unique_ptr<Scene>
MakeScene(
    Scene::Kind kind,
    topo::IInstance *renderer,
    Trigen_Session simulator);
