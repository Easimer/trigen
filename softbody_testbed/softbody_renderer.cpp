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
    Command_Render_Particles(Softbody_Simulation* sim, Softbody_Render_Parameters const* params) : sim(sim), params(params) {}
private:
    Softbody_Render_Parameters const* params;
    Softbody_Simulation* sim;
    virtual void execute(gfx::IRenderer* renderer) override {
        std::vector<Vec3> lines;
        std::vector<Vec3> positions;
        std::vector<Vec3> predicted_positions;
        std::vector<Vec3> goal_positions;
        std::vector<Vec3> centers_of_masses;
        std::vector<Vec3> sizes;
        std::vector<Quat> rotations;

        std::vector<Vec3> sizes_virtual;

        auto iter = sb::get_particles(sim);
        while(!iter->ended()) {
            auto particle = iter->get();
            lines.push_back(particle.start);
            lines.push_back(particle.end);

            positions.push_back(particle.position);
            sizes.push_back(particle.size);
            rotations.push_back(particle.orientation);
            sizes_virtual.push_back(Vec3(0.25, 0.25, 0.25));

            iter->step();
        }
        iter->release();

        iter = sb::get_particles_with_goal_position(sim);
        while (!iter->ended()) {
            auto particle = iter->get();
            goal_positions.push_back(particle.position);
            iter->step();
        }
        iter->release();

        for (iter = sb::get_centers_of_masses(sim); !iter->ended(); iter->step()) {
            auto particle = iter->get();
            centers_of_masses.push_back(particle.position);
        }
        iter->release();

        for (iter = sb::get_particles_with_predicted_position(sim); !iter->ended(); iter->step()) {
            auto particle = iter->get();
            predicted_positions.push_back(particle.position);
        }
        iter->release();

        assert(positions.size() == goal_positions.size());

        // TODO(danielm): we need a way to get back the sun position from
        // either the simulation or the application
        gfx::Render_Context_Supplement ctx;
        ctx.sun = params->sun_position;

        renderer->draw_lines(lines.data(), lines.size() / 2, Vec3(0, 0, 0), Vec3(0, 0.50, 0), Vec3(0, 1.00, 0));
        if (params->show_positions) {
            renderer->draw_ellipsoids(ctx, positions.size(), positions.data(), sizes.data(), rotations.data());
        }
        if (params->show_positions) {
            renderer->draw_ellipsoids(ctx, predicted_positions.size(), positions.data(), sizes.data(), rotations.data());
        }
        renderer->draw_ellipsoids(ctx, positions.size(), goal_positions.data(), sizes_virtual.data(), rotations.data(), Vec3(0.1, 0.8, 0.1));
        renderer->draw_ellipsoids(ctx, positions.size(), centers_of_masses.data(), sizes_virtual.data(), rotations.data(), Vec3(0.8, 0.1, 0.1));
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

class Render_Grid : public gfx::IRender_Command {
public:
    Render_Grid(Softbody_Simulation*) {}
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

        // render grid
        Vec3 grid[80];
        for (int i = 0; i < 20; i++) {
            auto base = 4 * i;
            grid[base + 0] = Vec3(i - 10, 0, -10);
            grid[base + 1] = Vec3(i - 10, 0, +10);
            grid[base + 2] = Vec3(-10, 0, i - 10);
            grid[base + 3] = Vec3(+10, 0, i - 10);
        }

        renderer->draw_lines(grid, 40, Vec3(0, 0, 0), Vec3(0.4, 0.4, 0.4), Vec3(0.4, 0.4, 0.4));
    }
};

class Visualize_Connections : public gfx::IRender_Command {
public:
    Visualize_Connections(Softbody_Simulation* s) : sim(s) {}
private:
    Softbody_Simulation* sim;
    virtual void execute(gfx::IRenderer* renderer) override {
        sb::Relation_Iterator* iter;
        std::vector<glm::vec3> lines;

        for (iter = sb::get_connections(sim); !iter->ended(); iter->step()) {
            auto rel = iter->get();
            lines.push_back(rel.parent_position);
            lines.push_back(rel.child_position);
        }
        iter->release();

        for (iter = sb::get_predicted_connections(sim); !iter->ended(); iter->step()) {
            auto rel = iter->get();
            lines.push_back(rel.parent_position);
            lines.push_back(rel.child_position);
        }
        iter->release();


        renderer->draw_lines(lines.data(), lines.size() / 2, Vec3(0, 0, 0), Vec3(.35, 0, 0), Vec3(1, 0, 0));
    }

};

template<typename T, class ... Arg>
static T* allocate_command_and_initialize(gfx::Render_Queue* rq, Arg ... args) {
    auto cmd = rq->allocate<T>();
    new(cmd) T(args...);
    rq->push(cmd);
    return cmd;
}

bool render_softbody_simulation(gfx::Render_Queue* rq, Softbody_Simulation* sim, Softbody_Render_Parameters const& params) {
    assert(rq != NULL);
    assert(sim != NULL);

    allocate_command_and_initialize<Render_Grid>(rq, sim);
    auto render_points = allocate_command_and_initialize<Command_Render_Points>(rq, sim);
    // auto render_apical_branches = allocate_command_and_initialize<Command_Render_Apical_Relations>(rq, sim);
    // auto render_lateral_branches = allocate_command_and_initialize<Command_Render_Lateral_Relations>(rq, sim);
    auto render_particles = allocate_command_and_initialize<Command_Render_Particles>(rq, sim, &params);
    allocate_command_and_initialize<Visualize_Connections>(rq, sim);

    return true;
}
