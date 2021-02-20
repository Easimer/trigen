// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <r_queue.h>
#include <marching_cubes.h>

class ISession {
public:
    virtual void render(gfx::Render_Queue *rq) = 0;

    virtual marching_cubes::params &marching_cubes_params() = 0;

    virtual void do_generate_mesh() = 0;
    virtual void do_paint_mesh() = 0;

    virtual char const *title() const = 0;
};

std::unique_ptr<ISession> make_session(char const *path_simulation_image);
