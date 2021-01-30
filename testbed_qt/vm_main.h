// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: main window viewmodel declaration
//

#pragma once

#include <softbody.h>

class IViewmodel_Main {
public:
    virtual ~IViewmodel_Main() = default;

    virtual bool add_mesh_collider(char const *path, std::string &err_msg) = 0;

    virtual int num_mesh_colliders() = 0;
    virtual bool mesh_collider(sb::Mesh_Collider *mesh, int index) = 0;
};

Unique_Ptr<IViewmodel_Main> make_viewmodel_main(sb::ISoftbody_Simulation *simulation);
