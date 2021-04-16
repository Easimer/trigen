// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <softbody.h>
#include <r_renderer.h>

class IMesh_Collider {
public:
    virtual ~IMesh_Collider() = default;
    virtual sb::ISoftbody_Simulation::Collider_Handle uploadToSimulation(sb::ISoftbody_Simulation *sim) = 0;
    virtual gfx::Model_ID uploadToRenderer(gfx::IRenderer *renderer) = 0;
    virtual gfx::Transform transform() const = 0;
};
