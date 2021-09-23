// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "scene.h"

class Scene_Basic_Cube : public Scene {
public:
    ~Scene_Basic_Cube() override = default;

    Scene_Basic_Cube(topo::IInstance *renderer, Trigen_Session simulation) {
        Trigen_Transform transform = { {}, { 1, 0, 0, 0 }, { 1, 1, 1 } };
         LoadObjMeshCollider(renderer, simulation, transform, "rockwall.obj");
    }
};

std::unique_ptr<Scene>
MakeScene_Basic_Cube(topo::IInstance *renderer, Trigen_Session simulation) {
    return std::make_unique<Scene_Basic_Cube>(renderer, simulation);
}
