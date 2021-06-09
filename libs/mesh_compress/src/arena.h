// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: arena container
//

#pragma once

#include <cstdint>
#include <vector>

class Arena {
public:
    void *allocate(size_t size);

    template<typename T>
    T* allocate() {
        return static_cast<T*>(allocate(sizeof(T)));
    }

    void const *data() const;
    size_t size() const;

private:
    std::vector<uint8_t> _backing;
};
