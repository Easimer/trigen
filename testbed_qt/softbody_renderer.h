// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#pragma once

#include "r_queue.h"
#include "softbody.h"
#include <r_cmd/softbody.h>

bool render_softbody_simulation(gfx::Render_Queue* rq, sb::ISoftbody_Simulation* sim, Softbody_Render_Parameters const& params);
void render_mesh_collider(gfx::Render_Queue *rq, sb::Mesh_Collider const *mesh);
