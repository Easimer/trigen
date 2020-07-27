// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#pragma once

#include "r_queue.h"
#include "softbody.h"

struct Softbody_Render_Parameters {
    Vec3 sun_position;
    bool draw_positions;
    bool draw_center_of_mass;
    bool draw_goal_position;
};

bool render_softbody_simulation(gfx::Render_Queue* rq, Softbody_Simulation* sim, Softbody_Render_Parameters const& params);
