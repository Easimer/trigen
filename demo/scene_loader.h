// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#pragma once

#include <string>

#include "scene.h"

bool
LoadSceneFromFile(
    std::string const &path,
    Scene &scene,
    topo::IInstance *renderer,
    Trigen_Session *session,
    std::vector<Scene::Collider> &colliders);

class Scene_Loader_Exception : public std::exception {
public:
    Scene_Loader_Exception(std::string msg)
        : std::exception(), _msg(std::move(msg)) { }

    Scene_Loader_Exception(Trigen_Status status)
        : std::exception()
        , _msg("Trigen status was not OK: " + std::to_string(status)) { }

    const char *
    what() const override {
        return _msg.c_str();
    }

    private:
    std::string _msg;
};
