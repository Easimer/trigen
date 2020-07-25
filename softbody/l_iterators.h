// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: iterators
//

#pragma once

#include <numeric>

#include <type_traits>
#include <iterator>

/**
 * Sum the values between two iterators.
 * A shorthand for std::accumulate where the initial value is zero.
 *
 * @param <T> Value type
 * @param <Iterator> Iterator type
 * @param first The iterator pointing to the first element
 * @param last The iterator pointing to the element after the last element
 * @return The sum
 */
template<typename T, typename Iterator>
T sum(Iterator first, Iterator last) {
    return std::accumulate(first, last, (T)0);
}

/**
 * Integer range
 *
 * Can be used to create an iterator that goes from integers A to B.
 */
class range {
public:
    class iterator {
        friend class range;
    public:
        constexpr long int operator*() const {
            return i_;
        }

        constexpr const iterator& operator++() {
            ++i_;
            return *this;
        }

        constexpr iterator operator++(int) {
            iterator copy(*this);
            ++i_;
            return copy;
        }

        constexpr bool operator==(const iterator& other) const {
            return i_ == other.i_;
        }

        constexpr bool operator!=(const iterator& other) const {
            return i_ != other.i_;
        }

        constexpr iterator() : i_(0) {}
    protected:
        constexpr iterator(long int start) : i_(start) {}

    private:
        unsigned long i_;
    };

    constexpr iterator begin() const {
        return begin_;
    }

    constexpr iterator end() const {
        return end_;
    }

    constexpr range(long int begin, long int end) : begin_(begin), end_(end) {}
private:
    iterator begin_;
    iterator end_;
};

// Iterator traits for range::iterator
template<>
struct std::iterator_traits<range::iterator> {
    using difference_type = long int;
    using value_type = long int;
    using pointer = range::iterator*;
    using reference = range::iterator&;
    using iterator_category = std::forward_iterator_tag;
};

/**
 * Can be used to concatenate two iterators (of the same value type) together
 * into one iterator.
 *
 * Since this is an iterator itself, this can be done recursively.
 */
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
