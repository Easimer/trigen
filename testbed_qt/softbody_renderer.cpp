// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#include "common.h"
#include <array>
#include <vector>
#include "softbody_renderer.h"
#include <r_cmd/general.h>
#include <r_cmd/softbody.h>

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
