// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: meshgen window implementation
//

#include "common.h"
#include "wnd_meshgen.h"
#include "softbody_renderer.h"

#include <trigen/tree_meshifier.h>

#include <dual_contouring.h>

using namespace std::placeholders;

Window_Meshgen::Window_Meshgen(
    sb::Unique_Ptr<sb::ISoftbody_Simulation>& simulation,
    QMainWindow* parent) :
    QDialog(parent),
    simulation(simulation) {
    update_mesh();
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

        mesh = ProcessTree(tree, [](auto, auto, auto, auto, auto, auto) { return 0.25f; });
    }
}

class Render_Generated_Mesh : public gfx::IRender_Command {
public:
    Render_Generated_Mesh(Mesh_Builder::Optimized_Mesh const& mesh) : mesh(mesh) {}

private:
    Mesh_Builder::Optimized_Mesh const& mesh;

    void execute(gfx::IRenderer* renderer) override {
        renderer->draw_triangle_elements(mesh.VerticesSize(), mesh.vertices.data(), mesh.ElementsSize(), mesh.elements.data(), Vec3());
    }
};

void Window_Meshgen::render(gfx::Render_Queue* rq) {
    auto c = rq->allocate<Render_Generated_Mesh>();
    new(c) Render_Generated_Mesh(mesh);
    rq->push(c);
}
