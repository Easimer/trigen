// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <optional>

#include <trigen.hpp>

class IMesh_Collider {
public:
    virtual ~IMesh_Collider() = default;
    virtual std::optional<trigen::Collider> uploadToSimulation(trigen::Session &sim) = 0;
    virtual topo::Model_ID uploadToRenderer(topo::IInstance *renderer) = 0;
    virtual topo::Transform transform() const = 0;
};
