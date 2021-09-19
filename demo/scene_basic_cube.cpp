// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "scene.h"

class Scene_Basic_Cube : public Scene {
public:
    ~Scene_Basic_Cube() override = default;

    Scene_Basic_Cube(topo::IInstance *renderer) {
        _models = LoadObjMesh(renderer, "cube.obj");

        topo::Material_ID material = nullptr;
        if (!renderer->CreateSolidColorMaterial(&material, { 0.9, 0.9, 0.9 })) {
            return;
        }

        _materials.push_back(material);

        for (auto &model : _models) {
            topo::Renderable_ID renderable = nullptr;
            renderer->CreateRenderable(&renderable, model, material);
            _renderables.push_back(renderable);
        }
    }

    void
    Cleanup(topo::IInstance *renderer) override {
        for (auto &id : _models) {
            renderer->DestroyModel(id);
        }
    }

    void
    Render(topo::IRender_Queue* rq) override {
        topo::Transform origin = { { 0, 0, 0 }, { 1, 0, 0, 0 }, { 1, 1, 1 } };
        for (auto &renderable : _renderables) {
            rq->Submit(renderable, origin);
        }
    }

private:
    std::vector<topo::Model_ID> _models;
    std::vector<topo::Material_ID> _materials;
    std::vector<topo::Renderable_ID> _renderables;
};

std::unique_ptr<Scene>
MakeScene_Basic_Cube(topo::IInstance *renderer) {
    return std::make_unique<Scene_Basic_Cube>(renderer);
}
