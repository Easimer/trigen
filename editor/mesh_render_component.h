// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <topo.h>

struct Mesh_Render_Component {
    topo::Renderable_ID renderable;
    topo::Model_ID model;
    topo::Material_ID material;
};
