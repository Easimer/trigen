// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: intersection subroutines
//

#pragma once
#include <glm/vec3.hpp>

namespace intersect {
    // Ray-triangle intersection
    // Returns true on intersection and fills in `xp` and `t`, such that
    // `origin + t * dir = xp`.
    // Returns false otherwise.
    bool ray_triangle(
        glm::vec3 &xp, float &t,
        glm::vec3 const &origin, glm::vec3 const &dir,
        glm::vec3 const &v0, glm::vec3 const &v1, glm::vec3 const &v2
    );

    // Ray-AABB intersection
    // Returns whether the two entities intersect.
    // The ray direction vector MUST be inverted component-wise,
    // that is, if the direction is (x, y, z), dir_inv is (1/x, 1/y, 1/z).
    bool ray_aabb(
        glm::vec3 const &origin, glm::vec3 const &dir_inv,
        glm::vec3 const &min, glm::vec3 const &max
    );
}
