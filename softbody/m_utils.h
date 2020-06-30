// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: math utils
//

#pragma once

#include "common.h"
#include <numeric>

template<typename T, typename Iterator>
T sum(Iterator begin, Iterator end) {
    return std::accumulate(begin, end, (T)0);
}

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
 * Perform a polar decomposition on a matrix and return the rotation matrix.
 * @param A A square matrix
 * @return The rotation matrix
*/
Mat3 polar_decompose_r(Mat3 const& A);

/**
 * Integer range
 */
class range {
public:
    class iterator {
        friend class range;
    public:
        constexpr long int operator *() const { return i_; }
        constexpr const iterator& operator ++() { ++i_; return *this; }
        constexpr iterator operator ++(int) { iterator copy(*this); ++i_; return copy; }

        constexpr bool operator ==(const iterator& other) const { return i_ == other.i_; }
        constexpr bool operator !=(const iterator& other) const { return i_ != other.i_; }

        constexpr iterator() : i_(0) {}
    protected:
        constexpr iterator(long int start) : i_(start) {}

    private:
        unsigned long i_;
    };

    constexpr iterator begin() const { return begin_; }
    constexpr iterator end() const { return end_; }
    constexpr range(long int  begin, long int end) : begin_(begin), end_(end) {}
private:
    iterator begin_;
    iterator end_;
};

template<>
struct std::iterator_traits<range::iterator> {
    using difference_type = long int;
    using value_type = long int;
    using pointer = range::iterator*;
    using reference = range::iterator&;
    using iterator_category = std::forward_iterator_tag;
};
