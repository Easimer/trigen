// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <optional>

#include <r_renderer.h>
#include <trigen.hpp>

class IMesh_Collider {
public:
    virtual ~IMesh_Collider() = default;
    virtual std::optional<trigen::Collider> uploadToSimulation(trigen::Session &sim) = 0;
    virtual gfx::Model_ID uploadToRenderer(gfx::IRenderer *renderer) = 0;
    virtual gfx::Transform transform() const = 0;
};
