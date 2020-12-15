// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: intersection algorithms
//

#include "intersection.h"

namespace intersect {
    bool intersect_ray_triangle(
        glm::vec3& xp, float& t,
        glm::vec3 const& origin, glm::vec3 const& dir,
        glm::vec3 const& v0, glm::vec3 const& v1, glm::vec3 const& v2
    ) {
        auto edge1 = v2 - v0;
        auto edge0 = v1 - v0;
        auto h = cross(dir, edge1);
        auto a = dot(edge0, h);
        if (-glm::epsilon<float>() < a && a < glm::epsilon<float>()) {
            return false;
        }

        auto f = 1.0f / a;
        auto s = origin - v0;
        auto u = f * dot(s, h);

        if (u < 0 || 1 < u) {
            return false;
        }

        auto q = cross(s, edge0);
        auto v = f * dot(dir, q);

        if (v < 0 || u + v > 1) {
            return false;
        }

        t = f * dot(edge1, q);
        if (t <= glm::epsilon<float>()) {
            return false;
        }

        xp = origin + t * dir;
        return true;
    }

    bool intersect_ray_aabb(
        glm::vec3 const& origin, glm::vec3 const& dir_inv,
        glm::vec3 const& min, glm::vec3 const& max
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