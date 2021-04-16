// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <r_renderer.h>

struct Mesh_Render_Component {
    gfx::Model_ID model;
    gfx::Material_Unlit material;
};

struct Untextured_Mesh_Render_Component {
    gfx::Model_ID model;
};
