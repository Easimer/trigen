// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#pragma once

#include <string>

#include "scene.h"

struct Demo {
    enum {
        NONE,
        ONESHOT,
        TIMELAPSE,
    } kind;

    union {
        struct {
            float at;
            float hold;
        } oneshot;
        struct {
            float from;
            float to;
            float step;
            float stepFrequency;
        } timelapse;
    };
};

bool
LoadSceneFromFile(
    std::string const &path,
    Scene &scene,
    topo::IInstance *renderer,
    Trigen_Session *session,
    std::vector<Scene::Collider> &colliders,
    Demo &demo);

class Scene_Loader_Exception : public std::exception {
public:
    Scene_Loader_Exception(std::string msg)
        : std::exception(), _msg(std::move(msg)) { }

    Scene_Loader_Exception(Trigen_Status status)
        : std::exception()
        , _msg("Trigen status was not OK: " + std::to_string(status)) { }

    const char *
    what() const noexcept override {
        return _msg.c_str();
    }

    private:
    std::string _msg;
};
