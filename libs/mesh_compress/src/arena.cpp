// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: arena container
//

#include "stdafx.h"

#include "arena.h"

void *Arena::allocate(size_t size) {
    _backing.resize(_backing.size() + size);
    return &_backing.back() - size + 1;
}

void const *Arena::data() const {
    return _backing.data();
}

size_t Arena::size() const {
    return _backing.size();
}
