// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: meshgen window implementation
//

#include "common.h"
#include "wnd_meshgen.h"
#include "softbody_renderer.h"

#include <glm/common.hpp>
#include <glm/vec3.hpp>

#include <marching_cubes.h>

#include <trigen/meshbuilder.h>
#include <psp/psp.h>

struct Generated_Mesh {
    size_t vertex_count, element_count;

    std::unique_ptr<std::array<float, 3>[]> position;
    std::unique_ptr<glm::vec2[]> uv;
    std::unique_ptr<glm::u8vec3[]> charter_debug_colors;

    std::unique_ptr<unsigned[]> element_indices;
};

class Window_Meshgen : public QDialog {
    Q_OBJECT;
public:
    ~Window_Meshgen() override = default;

    Window_Meshgen(
        sb::Unique_Ptr<sb::ISoftbody_Simulation>& simulation,
        QMainWindow* parent = nullptr
    );

public slots:
    void render(gfx::Render_Queue* rq);
    void update_mesh();

private:
    QHBoxLayout layout;
    sb::Unique_Ptr<sb::ISoftbody_Simulation>& simulation;
    Generated_Mesh mesh;
};

Window_Meshgen::Window_Meshgen(
    sb::Unique_Ptr<sb::ISoftbody_Simulation>& simulation,
    QMainWindow* parent) :
    QDialog(parent),
    simulation(simulation) {
    update_mesh();
}

static Generated_Mesh unpack_mesh(PSP::Mesh const& orig) {
    Generated_Mesh ret;

    ret.element_count = orig.elements.size();
    ret.vertex_count = orig.position.size();

    ret.position = std::make_unique<std::array<float, 3>[]>(orig.position.size());
    ret.uv = std::make_unique<glm::vec2[]>(orig.uv.size());
    ret.element_indices = std::make_unique<unsigned[]>(orig.elements.size());
    ret.charter_debug_colors = std::make_unique<glm::u8vec3[]>(orig.position.size());

    for (size_t i = 0; i < orig.position.size(); i++) {
        auto &pos = orig.position[i];
        auto &uv = orig.uv[i];

        ret.position[i] = { pos[0], pos[1], pos[2] };
        ret.uv[i] = { uv[0], uv[1] };
        ret.charter_debug_colors[i] = orig.chart_debug_color[i];
    }

    for (size_t i = 0; i < orig.elements.size(); i++) {
        ret.element_indices[i] = orig.elements[i];
    }

    return ret;
}

void Window_Meshgen::update_mesh() {
    auto plant_sim = simulation->get_extension_plant_simulation();

    if (plant_sim != nullptr) {
        std::vector<marching_cubes::metaball> metaballs;

        for (auto iter = simulation->get_particles(); !iter->ended(); iter->step()) {
            auto p = iter->get();
            metaballs.push_back({ p.position, (float)p.size.length() / 2 });
        }
        marching_cubes::params params;
        params.subdivisions = 16; // TODO: UI
        auto mesh = marching_cubes::generate(metaballs, params);
        printf("update_mesh: %zu triangles generated\n", mesh.indices.size() / 3);

        PSP::Material mat;
        PSP::Mesh pspMesh;
        pspMesh.position = mesh.positions;
        pspMesh.normal = mesh.normal;
        std::transform(mesh.indices.begin(), mesh.indices.end(), std::back_inserter(pspMesh.elements), [&](unsigned idx) { return (size_t)idx; });
        PSP::paint(mat, pspMesh);
        this->mesh = unpack_mesh(pspMesh);
    }
}

class Render_Generated_Mesh : public gfx::IRender_Command {
public:
    Render_Generated_Mesh(Generated_Mesh const& mesh) : mesh(mesh) {}

private:
    Generated_Mesh const& mesh;

    void execute(gfx::IRenderer* renderer) override {
        renderer->draw_triangle_elements(mesh.vertex_count, mesh.position.get(), mesh.element_count, mesh.element_indices.get(), Vec3());
    }
};

class Render_Generated_Mesh_With_Charter_Debug_Colors : public gfx::IRender_Command {
public:
    Render_Generated_Mesh_With_Charter_Debug_Colors(Generated_Mesh const &mesh) : mesh(mesh) {}
private:
    Generated_Mesh const &mesh;

    void execute(gfx::IRenderer* renderer) override {
        renderer->draw_triangle_elements_with_vertex_color(mesh.vertex_count, mesh.position.get(), mesh.charter_debug_colors.get(), mesh.element_count, mesh.element_indices.get(), Vec3());
    }
};

void Window_Meshgen::render(gfx::Render_Queue* rq) {
    // auto c = rq->allocate<Render_Generated_Mesh>();
    // new(c) Render_Generated_Mesh(mesh);
    auto c = rq->allocate<Render_Generated_Mesh_With_Charter_Debug_Colors>();
    new(c) Render_Generated_Mesh_With_Charter_Debug_Colors(mesh);
    rq->push(c);
}

QDialog *make_meshgen_window(sb::Unique_Ptr<sb::ISoftbody_Simulation> &simulation, QMainWindow *parent) {
    return new Window_Meshgen(simulation, parent);
}

#include "wnd_meshgen.moc"
