// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#include "stdafx.h"
#include "softbody_renderer.h"

class Command_Render_Points : public gfx::IRender_Command {
public:
    Command_Render_Points(Softbody_Simulation* sim) : sim(sim) {}
private:
    Softbody_Simulation* sim;
    virtual void execute(gfx::IRenderer* renderer) override {
        std::vector<Vec3> particles;
        auto iter = sb::get_particles(sim);

        while(!iter->ended()) {
            auto particle = iter->get();
            particles.push_back(particle.position);

            iter->step();
        }

        iter->release();

        renderer->draw_points(particles.data(), particles.size(), Vec3(0, 0, 0));
    }
};

class Command_Render_Apical_Relations : public gfx::IRender_Command {
public:
    Command_Render_Apical_Relations(Softbody_Simulation* sim) : sim(sim) {}
private:
    Softbody_Simulation* sim;
    virtual void execute(gfx::IRenderer* renderer) override {
        auto iter = sb::get_apical_relations(sim);
        std::vector<glm::vec3> lines;

        while(!iter->ended()) {
            auto rel = iter->get();
            lines.push_back(rel.parent_position);
            lines.push_back(rel.child_position);

            iter->step();
        }

        iter->release();

        auto col0 = Vec3(0, 0.5, 0);
        auto col1 = Vec3(0, 1.0, 0);
        renderer->draw_lines(lines.data(), lines.size() / 2, Vec3(0, 0, 0), col0, col1);

    }
};

class Command_Render_Lateral_Relations : public gfx::IRender_Command {
public:
    Command_Render_Lateral_Relations(Softbody_Simulation* sim) : sim(sim) {}
private:
    Softbody_Simulation* sim;
    virtual void execute(gfx::IRenderer* renderer) override {
        auto iter = sb::get_lateral_relations(sim);
        std::vector<glm::vec3> lines;

        while(!iter->ended()) {
            auto rel = iter->get();
            lines.push_back(rel.parent_position);
            lines.push_back(rel.child_position);

            iter->step();
        }

        iter->release();

        auto col0 = Vec3(0.5, 0, 0);
        auto col1 = Vec3(1.0, 0, 0);
        renderer->draw_lines(lines.data(), lines.size() / 2, Vec3(0, 0, 0), col0, col1);
    }
};

bool render_softbody_simulation(gfx::Render_Queue* rq, Softbody_Simulation* sim) {
    assert(rq != NULL);
    assert(sim != NULL);

    {
        auto cmd = rq->allocate<Command_Render_Points>();
        new(cmd) Command_Render_Points(sim);
        rq->push(cmd);
    }
    {
        auto cmd = rq->allocate<Command_Render_Apical_Relations>();
        new(cmd) Command_Render_Apical_Relations(sim);
        rq->push(cmd);
    }
    {
        auto cmd = rq->allocate<Command_Render_Lateral_Relations>();
        new(cmd) Command_Render_Lateral_Relations(sim);
        rq->push(cmd);
    }

    return true;
}
