// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include "types.h"

struct Collision_Constraint {
    unsigned pidx;
    Vec4 intersect, normal;
    float depth;
};