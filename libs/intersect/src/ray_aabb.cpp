// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include <intersect.h>

#include <glm/common.hpp>

namespace intersect {
    bool ray_aabb(
        glm::vec3 const &origin, glm::vec3 const &dir_inv,
        glm::vec3 const &min, glm::vec3 const &max
    ) {
        float tmin = -INFINITY, tmax = INFINITY;

        for (int i = 0; i < 3; i++) {
            auto t1 = (min[i] - origin[i]) * dir_inv[i];
            auto t2 = (max[i] - origin[i]) * dir_inv[i];

            tmin = glm::max(tmin, glm::min(t1, t2));
            tmax = glm::min(tmax, glm::max(t1, t2));
        }

        return tmax > glm::max(tmin, 0.0f);
    }
}
