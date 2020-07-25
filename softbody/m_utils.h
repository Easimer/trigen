// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: math utils
//

#pragma once

#include "common.h"
#include <numeric>

/**
 * Returns a vector such that if the biggest component of v is:
 * - X, then the return value is (1, 0, 0)
 * - Y, then the return value is (0, 1, 0)
 * - Z, then the return value is (0, 0, 1)
 *
 * @param v A positive vector
 */
Vec3 longest_axis_normalized(Vec3 const& v);

/**
* Compute the position of the head and tail of a particle given it's position,
* it's longest axis and it's orientation.
* @param pos Position of the particle
* @param longest_axis A vector describing the longest axis of the particle
* @param orientation Particle orientation
* @param out_head Pointer whither the head position will be written
* @param out_tail Pointer whither the tail position will be written
*/
void get_head_and_tail_of_particle(
    Vec3 const& pos,
    Vec3 const& longest_axis,
    Quat const& orientation,
    Vec3* out_head,
    Vec3* out_tail
);

/**
 * Extract the rotational part of an arbitrary matrix A.
 *
 * Implementation of "Matthias Mueller and Jan Bender and Nuttapong Chentanez
 * and Miles Macklin: A Robust Method to Extract the Rotational Part of
 * Deformations"
 * https://animation.rwth-aachen.de/publication/0548/
 *
 * @param A An arbitrary 3x3 transformation matrix
 * @param q A quaternion that approximates `A`
 */
void mueller_rotation_extraction(Mat3 const& A, Quat& q);
