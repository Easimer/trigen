// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: OpenGL utilities
//

#pragma once

#include "glres.h"

/**
 * VAO and VBO recycler for stream drawing.
 */
template<typename Tuple>
struct Array_Recycler {
public:
    /**
     * Get an unused instance of `Tuple`.
     * @param out Where the pointer to the tuple will be placed.
     * @return Handle to the instance.
     */
    size_t get(Tuple** out) {
        *out = NULL;
        if (ready_queue.empty()) {
            return make_new(out);
        } else {
            auto ret = ready_queue.front();
            ready_queue.pop();
            *out = &arrays[ret];
            return ret;
        }
    }

    /**
     * Mark a tuple instance as used and retire it.
     * @param handle An instance handle returned from get(Tuple**).
     */
    void put_back(size_t handle) {
        assert(handle < arrays.size());
        retired_queue.push(handle);
    }

    /**
     * Called after a frame ends.
     */
    void flip() {
        while (!retired_queue.empty()) {
            auto h = retired_queue.front();
            retired_queue.pop();
            ready_queue.push(h);
        }

        assert(retired_queue.size() == 0);
        assert(ready_queue.size() == arrays.size());
    }

    long long count() const {
        return retired_queue.size();
    }
protected:
    size_t make_new(Tuple** out) {
        auto ret = arrays.size();
        arrays.push_back(Tuple());
        *out = &arrays.back();
        return ret;
    }
private:
    std::queue<size_t> ready_queue;
    std::queue<size_t> retired_queue;
    std::vector<Tuple> arrays;
};

