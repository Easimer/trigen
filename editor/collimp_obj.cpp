// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include "collimp.h"
#include <tiny_obj_loader.h>

class Collider_Importer_OBJ : public ICollider_Importer {
public:
    ~Collider_Importer_OBJ() override = default;

    std::unique_ptr<IMesh_Collider> load_from_file(char const *path) override {
        // TODO(danielm): 
        return nullptr;
    }
};

std::unique_ptr<ICollider_Importer> make_obj_collider_importer() {
    auto importer = std::make_unique<Collider_Importer_OBJ>();

    return importer;
}
