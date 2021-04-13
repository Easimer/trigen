// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <r_renderer.h>
#include <r_queue.h>

#include <glm/vec3.hpp>

#include <softbody.h>

struct Softbody_Render_Parameters {
    glm::vec3 sun_position;
    bool draw_positions;
    bool draw_center_of_mass;
    bool draw_goal_position;
    bool draw_bind_pose;
};

class Render_Points : public gfx::IRender_Command {
public:
    Render_Points(sb::ISoftbody_Simulation *sim);
private:
    sb::ISoftbody_Simulation *sim;
    virtual void execute(gfx::IRenderer *renderer) override;
};

class Render_Particles : public gfx::IRender_Command {
public:
    Render_Particles(sb::ISoftbody_Simulation *sim, Softbody_Render_Parameters const &params);
private:
    Softbody_Render_Parameters const params;
    sb::ISoftbody_Simulation *sim;
    virtual void execute(gfx::IRenderer *renderer) override;
};

class Visualize_Connections : public gfx::IRender_Command {
public:
    Visualize_Connections(sb::ISoftbody_Simulation *s);
private:
    sb::ISoftbody_Simulation *sim;
    virtual void execute(gfx::IRenderer *renderer) override;
};

class Visualize_Bind_Pose : public gfx::IRender_Command {
public:
    Visualize_Bind_Pose(sb::ISoftbody_Simulation *s);
private:
    sb::ISoftbody_Simulation *sim;

    virtual void execute(gfx::IRenderer *renderer) override;
};

