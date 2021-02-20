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

class Session : public ISession {
public:
    Session(
        std::string title,
        std::vector<marching_cubes::metaball> &&metaballs
    ) : _title(std::move(title)), _metaballs(metaballs), _mc_params{}, _psp_mesh{} {
    }

    marching_cubes::params &marching_cubes_params() override { return _mc_params; }

    void render(gfx::Render_Queue *rq) override {
    }

    void do_generate_mesh() override {
        auto mesh = marching_cubes::generate(_metaballs, _mc_params);

        _psp_mesh.position = mesh.positions;
        _psp_mesh.normal = mesh.normal;

        _psp_mesh.elements.clear();
        std::transform(mesh.indices.begin(), mesh.indices.end(), std::back_inserter(_psp_mesh.elements), [&](unsigned idx) { return (size_t)idx; });
    }

    void do_paint_mesh() override {
    }

    char const *title() const override {
        return _title.c_str();
    }

private:
    std::string _title;
    std::vector<marching_cubes::metaball> _metaballs;

    marching_cubes::params _mc_params;
    PSP::Mesh _psp_mesh;
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
