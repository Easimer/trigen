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

    /*
     * Ray-sphere intersection test.
     *
     * On intersection the value of `xp` will be filled in.
     *
     * \param sphere_origin Sphere origin
     * \param sphere_radius Sphere radius
     * \param ray_origin Ray origin
     * \param ray_direction Ray direction
     * \param[out] xp Intersection point
     * \return A value indicating whether the ray intersects the sphere or not.
     */
    bool ray_sphere(
        glm::vec3 const &sphere_origin, float sphere_radius,
        glm::vec3 const &ray_origin, glm::vec3 const &ray_direction,
        glm::vec3 &xp
    );

    /*
     * \brief Ray-sphere intersection test.
     * 
     * `arr_ray_origin`, `arr_ray_direction`, `arr_ray_result` and
     * `arr_xp` must have enough space to hold `num_rays` elements.
     * 
     * \param sphere_origin Sphere origin
     * \param sphere_radius Sphere radius
     * \param num_rays Number of rays
     * \param arr_ray_origin Array of ray origins
     * \param arr_ray_direction Array of ray directions
     * \param arr_ray_result Array of results; a value of 1 meaning
     * that the ray intersected the sphere and a value of 0 meaning
     * the opposite.
     * \param arr_xp Array of intersection points.
     */
    void ray_sphere(
        glm::vec3 const &sphere_origin, float sphere_radius,
        size_t num_rays,
        glm::vec3 const *arr_ray_origin, glm::vec3 const *arr_ray_direction,
        int *arr_ray_result, glm::vec3 *arr_xp
    );
}
