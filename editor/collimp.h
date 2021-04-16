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

    virtual std::vector<std::unique_ptr<IMesh_Collider>> loadFromFile(char const *path) = 0;
};

std::unique_ptr<ICollider_Importer> makeObjColliderImporter();

inline std::unique_ptr<ICollider_Importer> makeColliderImporter(Mesh_Source_Kind kind) {
    switch (kind) {
    case Mesh_Source_Kind::OBJ:
        return makeObjColliderImporter();
    default:
        assert(!"Unhandled mesh source kind");
        break;
    }

    return nullptr;
}
