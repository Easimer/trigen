// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: raymarching utilities
//

#include "stdafx.h"
#include "m_sdf.h"

namespace sdf {
    Vec3 normal(Function const& f, Vec3 const& sp, float smoothness) {
        Vec3 n;
        auto xyy = Vec3(smoothness, 0, 0);
        auto yxy = Vec3(0, smoothness, 0);
        auto yyx = Vec3(0, 0, smoothness);
        n.x = f(sp + xyy) - f(sp - xyy);
        n.y = f(sp + yxy) - f(sp - yxy);
        n.z = f(sp + yyx) - f(sp - yyx);
        return glm::normalize(n);
    }

    Vec3 normal(Function const& f, Vec3 const& sp) {
        return normal(f, sp, 1.0f);
    }

    float raymarch(
        sdf::Function const& f,
        int steps,
        Vec3 start, Vec3 dir,
        float epsilon, float near_plane, float far_plane,
        std::function<void(float dist)> const& on_hit
    ) {
        float dist = 0;
        for (auto step = 0; step < steps; step++) {
            auto p = start + dist * dir;
            float temp = f(p);
            if (temp < 0.05) {
                break;
            }

            dist += temp;

            if (dist > 1) {
                break;
            }
        }

        if (near_plane <= dist && dist < far_plane) {
            on_hit(dist);
        }

        return dist;
    }
}
