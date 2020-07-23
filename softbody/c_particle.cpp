// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: math utils
//

#include "stdafx.h"
#include "common.h"
#include "c_particle.h"

namespace coll {
    static std::tuple<float, float> max(Vec3 const& s0, Vec3 const& s1) {
        auto idx0 =
            (s0.x > s0.y) ?
            ((s0.x > s0.z) ? (0) : (2))
            :
            ((s0.y > s0.z ? (1) : (2)));
        auto idx1 =
            (s1.x > s1.y) ?
            ((s1.x > s1.z) ? (0) : (2))
            :
            ((s1.y > s1.z ? (1) : (2)));

        return { s0[idx0], s1[idx1] };
    }

    std::optional<float> particle_particle_distance(
        Vec3 const& normal,
        Vec3 const& p0, Quat const& r0, Vec3 const& s0,
        Vec3 const& p1, Quat const& r1, Vec3 const& s1
    ) {
        // Assume that the particle are intersecting if the distance between
        // the centers of the two particles is less than the sum of the length
        // of their longest axes
        auto [axis0, axis1] = max(s0, s1);
        auto axis_sum = axis0 + axis1;
        auto axis_sum_sq = axis_sum * axis_sum;
        auto dir = p1 - p0;
        auto dist_sq = glm::dot(dir, dir);

        if (dist_sq > axis_sum_sq) {
            return std::nullopt;
        }

        // Expand normal to vec4 so that it can be multiplied by mat4's
        auto n = glm::vec4(normal, 1.0f);
        auto R_0 = glm::mat4(r0);
        auto R_1 = glm::mat4(r1);
        auto A_0 = R_0 * glm::scale(1.0f / s0) * glm::transpose(R_0);
        auto A_1 = R_1 * glm::scale(1.0f / s1) * glm::transpose(R_1);
        auto A_0_inv = R_0 * glm::scale(s0) * glm::transpose(R_0);
        auto A_1_inv = R_1 * glm::scale(s1) * glm::transpose(R_1);

        auto lambda = glm::sqrt(glm::dot(n, A_0 * A_1_inv * n));

        if (lambda < 0) {
            return std::nullopt;
        }

        auto B = glm::inverse(lambda * A_1 - A_0) * A_1;
        // 1 / (d**2)
        auto inv_d_sq = lambda * lambda * glm::dot(n, glm::transpose(B) * A_1 * B * n);
        auto d = glm::sqrt(1 / inv_d_sq);

        return d;
    }

    /**
     * Resolve collision between two particles given their positions, a normal
     * vector and a displacement value.
     * @param normal The displacement direction
     * @param p0 Position of the first particle (will not be moved)
     * @param p1 Position of the second particle (will be moved)
     * @param d Displacement
     */
    void resolve_particle_particle_collision(
        Vec3 const& normal, Vec3 const& p0, Vec3& p1, float d
    ) {
        // NOTE(danielm): this can be expressed in terms of FMA's
        p1 = p1 + d * normal;
    }
}
