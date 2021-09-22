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
            topo::Renderable_ID hVisual,
            Trigen_Collider hSimulation,
            topo::Transform const &transform)
            : hVisual(hVisual)
            , hSimulation(hSimulation)
            , transform(transform) { }
        topo::Renderable_ID hVisual;
        Trigen_Collider hSimulation;
        topo::Transform transform;
    };

    virtual ~Scene() = default;

    virtual void
    Cleanup(topo::IInstance *renderer);

    virtual void
    Render(topo::IRender_Queue *rq);

    virtual std::vector<Collider>
    LoadObjMeshCollider(
        topo::IInstance *renderer,
        Trigen_Session simulation,
        Trigen_Transform const &transform,
        char const *path);

    enum Kind {
        K_BASIC_CUBE,
    };

    private:
    std::vector<topo::Model_ID> _models;
    std::vector<topo::Texture_ID> _textures;
    std::vector<topo::Material_ID> _materials;

    std::vector<Collider> _environment;
};

std::unique_ptr<Scene>
MakeScene(
    Scene::Kind kind,
    topo::IInstance *renderer,
    Trigen_Session simulator);
