// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: raymarching utilities
//

#pragma once
#include "common.h"
#include "softbody.h"
#include <glm/vec3.hpp>

namespace sdf {
    using Function = sb::Signed_Distance_Function;

    Vec3 normal(Function const& f, Vec3 const& sp, float smoothness);
    Vec3 normal(Function const& f, Vec3 const& sp);

    float raymarch(
        sdf::Function const& f,
        int steps,
        Vec3 start, Vec3 dir,
        float epsilon, float near_plane, float far_plane,
        std::function<void(float dist)> const& on_hit
    );
}
