// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <cstdint>

#include <softbody.h>

enum class Collider_Handle_Kind {
    SDF = 0,
    Mesh,
    Max
};

static sb::ISoftbody_Simulation::Collider_Handle make_collider_handle(Collider_Handle_Kind kind, size_t index) {
    static_assert(sizeof(sb::ISoftbody_Simulation::Collider_Handle) == 8, "Size of a collider handle was not 8 bytes.");
    assert((index & 0xF000'0000'0000'0000) == 0);
    assert((int)kind < 256);
    auto kindNibble = (size_t)kind;
    // Encode the type of the handle in the most significant nibble
    return (index & 0x0FFF'FFFF'FFFF'FFFF) | (kindNibble << 60);
}

static void decode_collider_handle(sb::ISoftbody_Simulation::Collider_Handle handle, Collider_Handle_Kind &kind, size_t &index) {
    auto kindNibble = (handle >> 60) & 0x0F;
    assert(kindNibble < (int)Collider_Handle_Kind::Max);
    kind = (Collider_Handle_Kind)kindNibble;
    index = (handle & 0x0FFF'FFFF'FFFF'FFFF);
}

