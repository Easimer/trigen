// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: raymarching utilities
//

#pragma once

#define GLM_ENABLE_EXPERIMENTAL
#include <functional>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/vec_swizzle.hpp>

namespace sdf {
    /*
     * 3-dimensional signed distance function.
     * @param sp Sample point
     * @return The distance between the sample point and the surface of the shape.
     */
    using Function = std::function<float(glm::vec3 const& sp)>;

    /*
     * Calculate the surface normal of a shape at the sample point.
     * @param f Signed distance function
     * @param sp Sample point
     * @param smoothness Surface smoothness, [0.0, 1.0]
     * @return Surface normal.
     */
    inline glm::vec3 normal(Function const& f, glm::vec3 const& sp, float smoothness) {
        glm::vec3 n;
        auto xyy = glm::vec3(smoothness, 0, 0);
        auto yxy = glm::vec3(0, smoothness, 0);
        auto yyx = glm::vec3(0, 0, smoothness);
        n.x = f(sp + xyy) - f(sp - xyy);
        n.y = f(sp + yxy) - f(sp - yxy);
        n.z = f(sp + yyx) - f(sp - yyx);
        return normalize(n);
    }

    /*
     * Calculate the surface normal of a shape at the sample point.
     * This function assumes that the surface is completely smooth.
     * @param f Signed distance function
     * @param sp Sample point
     * @return Surface normal.
     */
    inline glm::vec3 normal(Function const& f, glm::vec3 const& sp) {
        return normal(f, sp, 1.0f);
    }

    /*
     * This function implements raymarching.
     * @param f Scene SDF
     * @param steps Maximum number of raymarching steps to do
     * @param start Origin of the ray
     * @param dir Direction of the ray
     * @param epsilon The distance threshold under which a sample point is
     * considered to be on the surface of the shape.
     * @param near_plane How far away is the near plane from the origin point.
     * @param far_plane How far away is the far plane from the origin point.
     * @param on_hit A function that will be called if the ray has hit
     * something. Its only argument is the distance from the starting point.
     * @return TODO(danielm):
     * 
     * @notes The purpose of the near_plane and far_plane parameters is to
     * terminate when the raymarching algorithm encounters degenerate cases:
     * - The ray goes too far away since there is no geometry present in the
     * direction it's going forward. The far_plane parameter can be used to
     * detect this state and return early.
     * - The surface of the shape is too close to the camera/origin point. The
     * near_plane parameter specifies what distance is "too close". Most of
     * the time this is only useful when rendering SDFs and this value should
     * be 0.
     * 
     * @todo What if the return type was std::optional<float>?
     */
    inline float raymarch(
        sdf::Function const& f,
        int steps,
        glm::vec3 start, glm::vec3 dir,
        float epsilon, float near_plane, float far_plane,
        std::function<void(float dist)> const& on_hit
    ) {
        float dist = 0;
        for (auto step = 0; step < steps; step++) {
            auto p = start + dist * dir;
            float temp = f(p);
            if (temp < epsilon) {
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

    /*
     * Signed distance function of a 3D sphere.
     * @param radius Sphere radius
     * @param sp Sample point
     * @return Distance between sample point and sphere surface
     */
    inline float sphere(float radius, glm::vec3 const& sp) {
        return length(sp) - radius;
    }

    /*
     * Signed distance function of a 3D plane.
     * @param n Plane normal
     * @param h Y-offset
     * @param sp Sample point
     * @return Distance between sample point and plane surface
     */
    inline float plane(glm::vec3 const& n, float h, glm::vec3 const& sp) {
        return dot(n, sp) + h;
    }

    /*
     * Signed distance function of a 3D box.
     * @param b Box extent
     * @param sp Sample point
     * @return Distance between sample point and box surface
     */
    inline float box(glm::vec3 const& b, glm::vec3 const& sp) {
        auto q = abs(sp) - b;
        return length(max(q, glm::vec3(0, 0, 0))) + glm::min(glm::max(q.x, glm::max(q.y, q.z)), 0.0f);
    }

    /*
     * Signed distance function of an infinite cylinder.
     * @param c A vector describing the cylinder. XY components are the center
     * of the cylinder, while the Z component is the cylinder radius.
     * @param sp Sample point
     * @return Distance between sample point and cylinder surface.
     *
     * @note The cylinder extends infinitely along the Y-axis.
     */
    inline float infCylinder(glm::vec3 const& c, glm::vec3 const& sp) {
        return length(glm::xz(sp) - glm::xy(c)) - c.z;
    }

    inline float translate(sdf::Function const& f, glm::vec3 const& origin, glm::vec3 const& sp) {
        return f(sp - origin);
    }

    inline Function translate(sdf::Function const& f, glm::vec3 const& origin) {
        return [f, origin](glm::vec3 const& sp) {
            return f(sp - origin);
        };
    }

    inline float rotate(sdf::Function const& f, glm::quat const& rotation, glm::vec3 const& sp) {
        auto q_inv = glm::conjugate(rotation);
        return f(q_inv * sp * rotation);
    }

    inline Function rotate(sdf::Function const& f, glm::quat const& rotation) {
        return [f, rotation](glm::vec3 const& sp) {
            auto q_inv = glm::conjugate(rotation);
            return f(q_inv * sp * rotation);
        };
    }
}