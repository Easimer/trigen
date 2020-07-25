// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: math utils
//

#pragma once

#include "common.h"
#include <numeric>

#include <type_traits>
#include <iterator>

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
[[deprecated]]
Mat3 polar_decompose_r(Mat3 const& A);

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

template<typename It1, typename It2>
class iterator_union {
private:
    using value_type1 = typename std::iterator_traits<It1>::value_type;
    using value_type2 = typename std::iterator_traits<It2>::value_type;
public:
    static_assert(std::is_same_v<value_type1, value_type2>, "Types must be the same!");
    static_assert(std::is_copy_assignable_v<It1>, "Iterator types must be copy assignable!");
    static_assert(std::is_copy_assignable_v<It2>, "Iterator types must be copy assignable!");
    class iterator {
    private:
        friend class iterator_union;
        It1 begin1, end1;
        It2 begin2, end2;
        bool first; // are we iterator the first iterator
    public:
        using difference_type = long int;
        using value_type = typename It1::value_type;
        using pointer = typename std::add_pointer<typename It1::value_type>::type;
        using reference = typename std::add_lvalue_reference<typename It1::value_type>::type;
        using iterator_category = std::forward_iterator_tag;

        reference operator*() {
            if (first) {
                return *begin1;
            } else {
                return *begin2;
            }
        }

        const iterator& operator++() {
            step();
            return *this;
        }

        iterator operator++(int) {
            iterator copy(*this);
            step();
            return copy;
        }

        bool operator==(iterator const& other) const {
            return first ? begin1 == other.begin1 : begin2 == other.begin2;
        }

        bool operator!=(iterator const& other) const {
            return first ? begin1 != other.begin1 : begin2 != other.begin2;
        }

    protected:
        iterator(It1 begin1, It1 end1, It2 begin2, It2 end2)
            : begin1(begin1), end1(end1), begin2(begin2), end2(end2), first(true) {}

        iterator(It1 begin1, It1 end1, It2 begin2, It2 end2, bool end)
            : begin1(end1), end1(end1), begin2(end2), end2(end2), first(false) {}

        void step() {
            if (first) {
                ++begin1;
                if (begin1 == end1) {
                    first = false;
                }
            } else {
                ++begin2;
            }
        }
    };

    iterator_union(It1 begin1, It1 end1, It2 begin2, It2 end2)
        : begin1(begin1), end1(end1), begin2(begin2), end2(end2) {}

    iterator begin() {
        return iterator(begin1, end1, begin2, end2);
    }

    iterator end() {
        return iterator(begin1, end1, begin2, end2, true);
    }

private:
    It1 begin1, end1;
    It2 begin2, end2;
};
