// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: math utils
//

#include "stdafx.h"
#include "m_utils.h"

Vec3 longest_axis_normalized(Vec3 const& v) {
    auto idx =
        (v.x > v.y) ?
        ((v.x > v.z) ? (0) : (2))
        :
        ((v.y > v.z ? (1) : (2)));
    Vec3 ret(0, 0, 0);
    ret[idx] = 1;
    return glm::normalize(ret);
}

void get_head_and_tail_of_particle(
    Vec3 const& pos,
    Vec3 const& longest_axis,
    Quat const& orientation,
    Vec3* out_head,
    Vec3* out_tail
) {
    auto const axis_rotated = orientation * longest_axis * glm::inverse(orientation);
    auto const axis_rotated_half = 0.5f * axis_rotated;
    *out_head = pos - axis_rotated_half;
    *out_tail = pos + axis_rotated_half;
}

Mat3 polar_decompose_r(Mat3 const& A) {
    glm::mat4 A4(A);
    glm::vec3 scale;
    glm::quat rotate;
    glm::vec3 translate;
    glm::vec3 skew;
    glm::vec4 perspective;
    auto v2 = A4[2];

    if (v2.z == 0.0f) {
        fprintf(stderr, "v2.z == 0.0f\n");
        v2.z = 1;
        A4[2] = v2;
    }

    A4[3] = glm::vec4(0, 0, 0, 1);
    if (glm::decompose(A4, scale, rotate, translate, skew, perspective)) {
        rotate = glm::conjugate(rotate);
        return (Mat3)rotate;
    } else {
        assert(0);
    }
}

#include <glm/gtc/matrix_access.hpp>
#define MUELLER2016_MAX_ITERATIONS (32)

void mueller_rotation_extraction(Mat3 const& A, Quat& q) {
    for (unsigned iter = 0; iter < MUELLER2016_MAX_ITERATIONS; iter++) {
        Mat3 R = Mat3(q);
        auto omega_v =
            glm::cross(glm::column(R, 0), glm::column(A, 0)) +
            glm::cross(glm::column(R, 1), glm::column(A, 1)) +
            glm::cross(glm::column(R, 2), glm::column(A, 2));
        auto omega_s =
            (1.0f / glm::abs(
                glm::dot(glm::column(R, 0), glm::column(A, 0)) +
                glm::dot(glm::column(R, 1), glm::column(A, 1)) +
                glm::dot(glm::column(R, 2), glm::column(A, 2))
            ) + 1.0e-9);

        auto omega = (float)omega_s * omega_v;

        auto w = glm::length(omega);
        if (w < 1.0e-9) {
            break;
        }

        q = glm::normalize(glm::angleAxis(w, (1.0f / w) * omega) * q);
    }
}
