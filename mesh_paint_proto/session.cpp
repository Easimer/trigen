// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"
#include "session.h"

#include <softbody.h>
#include <softbody/file_serializer.h>
#include <marching_cubes.h>
#include <psp/psp.h>

struct Debug_Mesh {
    std::vector<std::array<float, 3>> positions;
    std::vector<unsigned> elements;
};

struct Charter_Debug_Mesh : public Debug_Mesh {
    std::vector<glm::u8vec3> charter_debug_colors;
};

template<typename T>
class Render_Debug_Mesh : public gfx::IRender_Command {
public:
    Render_Debug_Mesh(T const *mesh) : _mesh(mesh) {
    }

    void execute(gfx::IRenderer *renderer) override {
        draw_mesh(renderer, *_mesh);
    }

    void draw_mesh(gfx::IRenderer *renderer, Charter_Debug_Mesh const &mesh) {
        auto vertex_count = mesh.positions.size();
        auto element_count = mesh.elements.size();

        renderer->draw_triangle_elements_with_vertex_color(
            vertex_count,
            mesh.positions.data(),
            mesh.charter_debug_colors.data(),
            element_count,
            mesh.elements.data(),
            glm::vec3()
        );
    }

    void draw_mesh(gfx::IRenderer *renderer, Debug_Mesh const &mesh) {
        auto vertex_count = mesh.positions.size();
        auto element_count = mesh.elements.size();

        renderer->draw_triangle_elements(
            vertex_count, 
            mesh.positions.data(),
            element_count,
            mesh.elements.data(),
            glm::vec3()
        );
    }
private:
    T const *_mesh;
};

class Session : public ISession {
public:
    Session(
        std::string title,
        std::vector<marching_cubes::metaball> &&metaballs
    ) : _title(std::move(title)), _metaballs(metaballs), _mc_params{}, _psp_mesh{} {
    }

    marching_cubes::params &marching_cubes_params() override { return _mc_params; }

    void render(gfx::Render_Queue *rq) override {
        if (_debug_mesh.has_value()) {
            gfx::allocate_command_and_initialize<Render_Debug_Mesh<Debug_Mesh>>(rq, &_debug_mesh.value());
        }
        if (_charter_debug_mesh.has_value()) {
            gfx::allocate_command_and_initialize<Render_Debug_Mesh<Charter_Debug_Mesh>>(rq, &_charter_debug_mesh.value());
        }
    }

    void do_generate_mesh() override {
        auto mesh = marching_cubes::generate(_metaballs, _mc_params);

        _psp_mesh.position = mesh.positions;
        _psp_mesh.normal = mesh.normal;

        _psp_mesh.elements.clear();
        std::transform(mesh.indices.begin(), mesh.indices.end(), std::back_inserter(_psp_mesh.elements), [&](unsigned idx) { return (size_t)idx; });

        _charter_debug_mesh.reset();
        _debug_mesh = convert_mesh(mesh);
    }

    void do_paint_mesh() override {
        int rc;
        // TODO(danielm): store this somewhere
        PSP::Material mat;
        rc = PSP::paint(mat, _psp_mesh);
        if (rc != 0) {
            return;
        }

        _debug_mesh.reset();
        _charter_debug_mesh = convert_mesh(_psp_mesh);

    }

    char const *title() const override {
        return _title.c_str();
    }

private:
    Debug_Mesh convert_mesh(marching_cubes::mesh const &mesh) {
        Debug_Mesh ret;

        std::transform(
            mesh.positions.begin(), mesh.positions.end(),
            std::back_inserter(ret.positions),
            [&](auto p) -> std::array<float, 3> {
                return { p.x, p.y, p.z };
            }
        );
        ret.elements = mesh.indices;
        return ret;
    }

    Charter_Debug_Mesh convert_mesh(PSP::Mesh const &mesh) {
        Charter_Debug_Mesh ret;

        std::transform(
            mesh.position.begin(), mesh.position.end(),
            std::back_inserter(ret.positions),
            [&](auto p) -> std::array<float, 3> {
                return { p.x, p.y, p.z };
            }
        );
        std::transform(
            mesh.elements.begin(), mesh.elements.end(),
            std::back_inserter(ret.elements),
            [&](auto p) { return (unsigned)p; }
        );

        ret.charter_debug_colors = mesh.chart_debug_color;

        return ret;
    }

private:
    std::string _title;
    std::vector<marching_cubes::metaball> _metaballs;

    marching_cubes::params _mc_params;
    PSP::Mesh _psp_mesh;
    std::optional<Debug_Mesh> _debug_mesh;
    std::optional<Charter_Debug_Mesh> _charter_debug_mesh;
};

std::unique_ptr<ISession> make_session(char const *path_simulation_image) {
    sb::Config cfg = {};
    cfg.compute_preference = sb::Compute_Preference::Reference;
    cfg.ext = sb::Extension::None;

    // Create a simulation so we can load the image
    auto simulation = sb::create_simulation(cfg);
    auto deserializer = sb::make_deserializer(path_simulation_image);
    if (!simulation->load_image(deserializer.get())) {
        return nullptr;
    }

    // Generate metaballs
    std::vector<marching_cubes::metaball> metaballs;

    for (auto iter = simulation->get_particles(); !iter->ended(); iter->step()) {
        auto p = iter->get();
        metaballs.push_back({ p.position, (float)p.size.length() / 2 });
    }

    return std::make_unique<Session>(
        path_simulation_image,
        std::move(metaballs)
    );
}
