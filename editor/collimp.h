// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: Collider import
//

#pragma once

#include <memory>

#include "mesh_collider.h"

enum class Mesh_Source_Kind {
    OBJ,
};

class ICollider_Importer {
public:
    virtual ~ICollider_Importer() = default;

    virtual std::unique_ptr<IMesh_Collider> load_from_file(char const *path) = 0;
};

std::unique_ptr<ICollider_Importer> make_obj_collider_importer();

inline std::unique_ptr<ICollider_Importer> make_collider_importer(Mesh_Source_Kind kind) {
    switch (kind) {
    case Mesh_Source_Kind::OBJ:
        return make_obj_collider_importer();
    default:
        assert(!"Unhandled mesh source kind");
        break;
    }
}
