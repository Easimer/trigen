// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#pragma once

#include <vector>

#include <topo.h>

namespace topo {
class Render_Queue : public IRender_Queue {
public:
    ~Render_Queue() override = default;

    void
    Submit(Renderable_ID renderable, Transform const &transform) override;

    void
    AddLight(
        glm::vec4 const &color,
        Transform const &transform,
        bool castsShadows) override;

    struct Command {
        Renderable_ID renderable;
        Transform transform;
    };

    struct Light {
        glm::vec4 light;
        Transform transform;
        bool castsShadows;
    };

    std::vector<Command> const &
    GetCommands() const {
        return _commands;
    }

    std::vector<Light> const &
    GetLights() const {
        return _lights;
    }

private:
    std::vector<Command> _commands;
    std::vector<Light> _lights;
};
}
