// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <r_queue.h>
#include <marching_cubes.h>
#include <psp/psp.h>

class ISession {
public:
    virtual ~ISession() = default;

    virtual void render(gfx::Render_Queue *rq) = 0;

    virtual marching_cubes::params &marching_cubes_params() = 0;
    virtual PSP::Parameters &paint_params() = 0;

    virtual void do_generate_mesh() = 0;
    virtual void do_unwrap_mesh() = 0;

    virtual void begin_painting() = 0;
    virtual void step_painting() = 0;
    virtual void stop_painting() = 0;

    virtual char const *title() const = 0;

    virtual PSP::Mesh *mesh() = 0;
    virtual PSP::Material *material() = 0;
};

std::unique_ptr<ISession> make_session(char const *path_simulation_image);
