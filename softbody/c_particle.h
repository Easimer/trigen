// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: particle-particle collision resolution
//

#pragma once

#include "common.h"
#include <glm/vec3.hpp>
#include <glm/gtc/quaternion.hpp>
#include <optional>

namespace coll {
    // TODO(danielm): rewrite these functions to process multiple particles
    // instead only of pairs of them.

    /**
     * Detect collision between two particles.
     *
     * NOTE: this won't resolve the collision.
     * See `coll::resolve_particle_particle_collision`.
     *
     * @param normal The displacement direction
     * @param p0 Position of the first particle
     * @param r0 Orientation of the first particle
     * @param s0 Radii of the first particle
     * @param p1 Position of the second particle
     * @param r1 Orientation of the second particle
     * @param s1 Radii of the second particle
     * @return The displacement parameter to pass to
     * `coll::resolve_particle_particle_collision` if the two particle collide,
     * or empty.
     */
    std::optional<float> particle_particle_distance(
        Vec3 const& normal,
        Vec3 const& p0, Quat const& r0, Vec3 const& s0,
        Vec3 const& p1, Quat const& r1, Vec3 const& s1
    );

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
    );
}
