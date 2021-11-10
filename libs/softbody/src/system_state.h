// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: common declarations
//

#pragma once

#include <softbody.h>
#include "types.h"
#include "collision_constraint.h"

struct Mesh_Collider_Slot {
    bool used;

    Mat4 transform;
    size_t triangle_count;
    Vector<uint64_t> vertex_indices;
    Vector<uint64_t> normal_indices;
    Vector<Vec3> vertices;
    Vector<Vec3> normals;

    Vec3 min, max;
};

struct SDF_Slot {
    bool used;
    sb::sdf::ast::Expression<float>* expr;
    sb::sdf::ast::Sample_Point* sp;
};

#include "system_state.generated.h"
