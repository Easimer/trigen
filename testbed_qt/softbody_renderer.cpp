// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#include "common.h"
#include <array>
#include <vector>
#include "softbody_renderer.h"

class Render_Points : public gfx::IRender_Command {
public:
    Render_Points(sb::ISoftbody_Simulation* sim) : sim(sim) {}
private:
    sb::ISoftbody_Simulation* sim;
    virtual void execute(gfx::IRenderer* renderer) override {
        std::vector<Vec3> particles;
        auto iter = sim->get_particles();

        while(!iter->ended()) {
            auto particle = iter->get();
            particles.push_back(particle.position);

            iter->step();
        }

        renderer->draw_points(particles.size(), particles.data(), Vec3(0, 0, 0));
    }
};

class Render_Particles : public gfx::IRender_Command {
public:
    Render_Particles(sb::ISoftbody_Simulation* sim, Softbody_Render_Parameters const& params) : sim(sim), params(params) {}
private:
    Softbody_Render_Parameters const params;
    sb::ISoftbody_Simulation* sim;
    virtual void execute(gfx::IRenderer* renderer) override {
        std::vector<Vec3> lines;
        std::vector<Vec3> positions;
        std::vector<Vec3> predicted_positions;
        std::vector<Vec3> goal_positions;
        std::vector<Vec3> centers_of_masses;
        std::vector<Vec3> sizes;
        std::vector<Quat> rotations;

        std::vector<Vec3> sizes_virtual;

        auto iter = sim->get_particles();
        while(!iter->ended()) {
            auto particle = iter->get();
            // lines.push_back(particle.start);
            // lines.push_back(particle.end);

            positions.push_back(particle.position);
            sizes.push_back(particle.size);
            rotations.push_back(particle.orientation);
            sizes_virtual.push_back(Vec3(0.25, 0.25, 0.25));

            iter->step();
        }

        for(iter = sim->get_particles_with_goal_positions(); !iter->ended(); iter->step()) {
            auto particle = iter->get();
            goal_positions.push_back(particle.position);
        }

        for (iter = sim->get_centers_of_masses(); !iter->ended(); iter->step()) {
            auto particle = iter->get();
            centers_of_masses.push_back(particle.position);
        }

        for (iter = sim->get_particles_with_predicted_positions(); !iter->ended(); iter->step()) {
            auto particle = iter->get();
            predicted_positions.push_back(particle.position);
        }

        assert(positions.size() == goal_positions.size());

        gfx::Render_Context_Supplement ctx;
        ctx.sun = params.sun_position;

        renderer->draw_lines(lines.data(), lines.size() / 2, Vec3(0, 0, 0), Vec3(0, 0.50, 0), Vec3(0, 1.00, 0));
        if (params.draw_positions) {
            renderer->draw_ellipsoids(ctx, positions.size(), positions.data(), sizes.data(), rotations.data());
            renderer->draw_ellipsoids(ctx, predicted_positions.size(), positions.data(), sizes.data(), rotations.data());
        }
        if (params.draw_goal_position) {
            renderer->draw_ellipsoids(ctx, positions.size(), goal_positions.data(), sizes_virtual.data(), rotations.data(), Vec3(0.1, 0.8, 0.1));
        }
        if (params.draw_center_of_mass) {
            renderer->draw_ellipsoids(ctx, positions.size(), centers_of_masses.data(), sizes_virtual.data(), rotations.data(), Vec3(0.8, 0.1, 0.1));
        }
    }
};

class Render_Grid : public gfx::IRender_Command {
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
    Visualize_Connections(sb::ISoftbody_Simulation* s) : sim(s) {}
private:
    sb::ISoftbody_Simulation* sim;
    virtual void execute(gfx::IRenderer* renderer) override {
        std::vector<glm::vec3> lines;
        sb::Unique_Ptr<sb::Relation_Iterator> iter;

        for (iter = sim->get_connections(); !iter->ended(); iter->step()) {
            auto rel = iter->get();

            lines.insert(lines.end(), { rel.parent_position, rel.child_position });
        }

        renderer->draw_lines(lines.data(), lines.size() / 2, Vec3(0, 0, 0), Vec3(.35, 0, 0), Vec3(1, 0, 0));
    }

};

class Visualize_Bind_Pose : public gfx::IRender_Command {
public:
    Visualize_Bind_Pose(sb::ISoftbody_Simulation* s) : sim(s) {}
private:
    sb::ISoftbody_Simulation* sim;

    virtual void execute(gfx::IRenderer* renderer) override {
        std::vector<glm::vec3> lines;
        std::vector<glm::vec3> particles;

        for (auto iter = sim->get_particles_in_bind_pose(); !iter->ended(); iter->step()) {
            auto particle = iter->get();
            particles.push_back(particle.position);
        }

        for (auto iter = sim->get_connections_in_bind_pose(); !iter->ended(); iter->step()) {
            auto rel = iter->get();
            lines.push_back(rel.parent_position);
            lines.push_back(rel.child_position);
        }

        renderer->draw_lines(lines.data(), lines.size() / 2, Vec3(0, 0, 0), Vec3(0, 0, 1), Vec3(0, 0, 1));
        renderer->draw_points(particles.size(), particles.data(), Vec3());
    }
};

bool render_softbody_simulation(gfx::Render_Queue* rq, sb::ISoftbody_Simulation* sim, Softbody_Render_Parameters const& params) {
    assert(rq != NULL);

    gfx::allocate_command_and_initialize<Render_Grid>(rq);

    if (sim != NULL) {
        gfx::allocate_command_and_initialize<Render_Points>(rq, sim);
        gfx::allocate_command_and_initialize<Render_Particles>(rq, sim, params);
        gfx::allocate_command_and_initialize<Visualize_Connections>(rq, sim);

        if (params.draw_bind_pose) {
            gfx::allocate_command_and_initialize<Visualize_Bind_Pose>(rq, sim);
        }
    }

    return true;
}

class Render_Mesh_Collider : public gfx::IRender_Command {
public:
    Render_Mesh_Collider(sb::Mesh_Collider const *mesh) {
        vertices.resize(mesh->position_count);
        auto positions = (std::array<float, 3> *)mesh->positions;

        for (size_t i = 0; i < mesh->position_count; i++) {
            vertices[i] = positions[i];
        }

        elements.resize(mesh->triangle_count * 3);
        for (size_t i = 0; i < mesh->triangle_count; i++) {
            elements[i * 3 + 0] = mesh->vertex_indices[i * 3 + 0];
            elements[i * 3 + 1] = mesh->vertex_indices[i * 3 + 1];
            elements[i * 3 + 2] = mesh->vertex_indices[i * 3 + 2];
        }
    }

    void execute(gfx::IRenderer *renderer) override {
        renderer->draw_triangle_elements(vertices.size(), vertices.data(), elements.size(), elements.data(), Vec3());
    }

    std::vector<std::array<float, 3>> vertices;
    std::vector<unsigned int> elements;
};

void render_mesh_collider(gfx::Render_Queue *rq, sb::Mesh_Collider const *mesh) {
    assert(rq != NULL);
    assert(mesh != NULL);

    gfx::allocate_command_and_initialize<Render_Mesh_Collider>(rq, mesh);
}
