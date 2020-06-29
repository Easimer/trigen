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

class Command_Render_Particles : public gfx::IRender_Command {
public:
    Command_Render_Particles(Softbody_Simulation* sim) : sim(sim) {}
private:
    Softbody_Simulation* sim;
    virtual void execute(gfx::IRenderer* renderer) override {
        std::vector<Vec3> lines;
        auto iter = sb::get_particles(sim);

        while(!iter->ended()) {
            auto particle = iter->get();
            lines.push_back(particle.start);
            lines.push_back(particle.end);

            iter->step();
        }

        iter->release();

        renderer->draw_lines(lines.data(), lines.size() / 2, Vec3(0, 0, 0), Vec3(0, 0.50, 0), Vec3(0, 1.00, 0));
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

class Render_Cube : public gfx::IRender_Command {
public:
    Render_Cube(Softbody_Simulation*) {}
private:
    virtual void execute(gfx::IRenderer* renderer) override {
        glm::vec3 lines[] = {
            glm::vec3(0, 0, 0),
            glm::vec3(1, 0, 0),
            glm::vec3(0, 0, 0),
            glm::vec3(0, 1, 0),
            glm::vec3(0, 0, 0),
            glm::vec3(0, 0, 1),
        };

        renderer->draw_lines(lines + 0, 1, Vec3(0, 0, 0), Vec3(.35, 0, 0), Vec3(1, 0, 0));
        renderer->draw_lines(lines + 2, 1, Vec3(0, 0, 0), Vec3(0, .35, 0), Vec3(0, 1, 0));
        renderer->draw_lines(lines + 4, 1, Vec3(0, 0, 0), Vec3(0, 0, .35), Vec3(0, 0, 1));
    }
};

template<typename T>
static T* allocate_command_and_initialize(gfx::Render_Queue* rq, Softbody_Simulation* sim) {
    auto cmd = rq->allocate<T>();
    new(cmd) T(sim);
    rq->push(cmd);
    return cmd;
}

bool render_softbody_simulation(gfx::Render_Queue* rq, Softbody_Simulation* sim) {
    assert(rq != NULL);
    assert(sim != NULL);

    allocate_command_and_initialize<Render_Cube>(rq, sim);
    auto render_points = allocate_command_and_initialize<Command_Render_Points>(rq, sim);
    // auto render_apical_branches = allocate_command_and_initialize<Command_Render_Apical_Relations>(rq, sim);
    // auto render_lateral_branches = allocate_command_and_initialize<Command_Render_Lateral_Relations>(rq, sim);
    auto render_particles = allocate_command_and_initialize<Command_Render_Particles>(rq, sim);

    return true;
}
