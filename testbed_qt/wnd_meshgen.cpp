// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: meshgen window implementation
//

#include "common.h"
#include "wnd_meshgen.h"
#include "softbody_renderer.h"

#include <trigen/tree_meshifier.h>

using namespace std::placeholders;

Window_Meshgen::Window_Meshgen(
    sb::Unique_Ptr<sb::ISoftbody_Simulation>& simulation,
    QMainWindow* parent) :
    QDialog(parent),
    simulation(simulation) {
    update_mesh();
}

static Generated_Mesh unpack_mesh(Optimized_Mesh<TG_Vertex> const& orig) {
    Generated_Mesh ret;

    ret.element_count = orig.ElementsCount();
    ret.vertex_count = orig.VerticesCount();

    ret.position = std::make_unique<std::array<float, 3>[]>(orig.VerticesCount());
    ret.uv = std::make_unique<glm::vec2[]>(orig.VerticesCount());
    ret.element_indices = std::make_unique<unsigned[]>(orig.ElementsCount());

    for (size_t i = 0; i < orig.VerticesCount(); i++) {
        auto& pos = orig.vertices[i].position;
        auto& uv = orig.vertices[i].uv;

        ret.position[i] = { pos[0], pos[1], pos[2] };
        ret.uv[i] = { uv[0], uv[1] };
    }

    for (size_t i = 0; i < orig.ElementsCount(); i++) {
        ret.element_indices[i] = orig.elements[i];
    }

    return ret;
}

void Window_Meshgen::update_mesh() {
    auto plant_sim = simulation->get_extension_plant_simulation();

    if (plant_sim != nullptr) {
        Tree_Node_Pool tree;

        for (auto iter = simulation->get_particles(); !iter->ended(); iter->step()) {
            /*
             * TODO(danielm): this algorithm relies on the fact that the iterator
             * will return particles in order, from idx 0 to idx N.
             */

            auto p = iter->get();
            uint32_t node_idx;
            auto& node = tree.Allocate(node_idx);

            assert(p.id == node_idx);

            node.vPosition = lm::Vector4(p.position.x, p.position.y, p.position.z);
            node.unUser = 0; // TODO(danielm): size
        }

        for (auto iter = plant_sim->get_parental_relations(); !iter->ended(); iter->step()) {
            auto rel = iter->get();
            auto& parent = tree.GetNode(rel.parent);
            parent.AddChild(rel.child);
        }

        mesh = unpack_mesh(ProcessTree(tree, [](auto, auto, auto, auto, auto, auto) { return 0.25f; }));
    }
}

class Render_Generated_Mesh : public gfx::IRender_Command {
public:
    Render_Generated_Mesh(Generated_Mesh const& mesh) : mesh(mesh) {}

private:
    Generated_Mesh const& mesh;

    void execute(gfx::IRenderer* renderer) override {
        renderer->draw_triangle_elements(mesh.vertex_count * sizeof(glm::vec3), mesh.position.get(), mesh.element_count * sizeof(unsigned), mesh.element_indices.get(), Vec3());
    }
};

void Window_Meshgen::render(gfx::Render_Queue* rq) {
    auto c = rq->allocate<Render_Generated_Mesh>();
    new(c) Render_Generated_Mesh(mesh);
    rq->push(c);
}
