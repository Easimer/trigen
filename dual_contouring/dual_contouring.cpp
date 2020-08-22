// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: dual contouring library implementation
//

#include "dual_contouring.h"
#include "octree.h"

enum class Corner : uint8_t {
    C_Z   = 0b10000000, // Front-left-bottom
    C_ZX  = 0b01000000, // Front-right-bottom
    C_ZY  = 0b00100000, // Front-left-top
    C_ZXY = 0b00010000, // Front-right-top

    C_    = 0b00001000, // Back-left-bottom
    C_X   = 0b00000100, // Back-right-bottom
    C_Y   = 0b00000010, // Back-left-top
    C_XY  = 0b00000001, // Back-right-top
};

struct Cell_Mask {
    uint8_t value = 0;

    Cell_Mask& operator|=(Corner corner) noexcept {
        value |= static_cast<uint8_t>(corner);
        return *this;
    }

    bool sign_changed() const noexcept {
        return value != 0;
    }
};

namespace dc {
    Mesh dual_contour(Parameters const& params, Volume_Function const& function) {
        return {};
    }
}
