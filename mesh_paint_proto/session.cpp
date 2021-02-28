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
#include <stb_image.h>

extern "C" {
    extern uint8_t const *test_grid_png;
    extern unsigned long long test_grid_png_len;
}

struct Debug_Mesh {
    gfx::Model_ID renderer_handle = nullptr;

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

class Render_Model : public gfx::IRender_Command {
public:
    Render_Model(gfx::Model_ID model, gfx::Texture_ID diffuse, gfx::Transform const &transform)
        : _model(model), _diffuse(diffuse), _transform(transform) {
    }

    void execute(gfx::IRenderer *renderer) override {
        gfx::Material_Unlit material;
        material.diffuse = _diffuse;

        renderer->draw_textured_triangle_elements(_model, material, _transform);
    }
private:
    gfx::Model_ID _model;
    gfx::Texture_ID _diffuse;
    gfx::Transform _transform;
};

class Load_Texture_Command : public gfx::IRender_Command {
public:
    Load_Texture_Command(std::optional<gfx::Texture_ID> *handle, void const *image, size_t image_len)
    : _handle(handle), _image(image), _image_len(image_len) {
    }

    void execute(gfx::IRenderer *renderer) override {
        if (!_handle->has_value()) {
            gfx::Texture_ID id;
            int width, height;
            int channels;
            auto data = stbi_load_from_memory((stbi_uc*)_image, _image_len, &width, &height, &channels, 3);
            if (renderer->upload_texture(&id, width, height, gfx::Texture_Format::RGB888, data)) {
                _handle->emplace(id);
            }
            stbi_image_free(data);
        }
    }

private:
    std::optional<gfx::Texture_ID> *_handle;
    void const *_image;
    size_t _image_len;
};

class Upload_Model_Command : public gfx::IRender_Command {
public:
    Upload_Model_Command(gfx::Model_ID *out_id, Charter_Debug_Mesh const *mesh) : _out_id(out_id), _mesh(mesh) {
    }

    void execute(gfx::IRenderer *renderer) override {
        gfx::Model_Descriptor model;

        model.vertex_count = _mesh->positions.size();
        model.vertices = _mesh->positions.data();
        model.element_count = _mesh->elements.size();
        model.elements = _mesh->elements.data();

        std::vector<glm::vec2> fake_uv;
        for (size_t i = 0; i < model.vertex_count; i++) {
            fake_uv.push_back({});
        }
        model.uv = (std::array<float, 2>*)fake_uv.data();

        renderer->create_model(_out_id, &model);
    }
private:
    gfx::Model_ID *_out_id;
    Charter_Debug_Mesh const *_mesh;
};

class Destroy_Model_Command : public gfx::IRender_Command {
public:
    Destroy_Model_Command(gfx::Model_ID id) : _id(id) {
    }

    void execute(gfx::IRenderer *renderer) override {
        renderer->destroy_model(_id);
    }
private:
    gfx::Model_ID _id;
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
        if (_charter_debug_mesh) {
            if (_charter_debug_mesh->renderer_handle == nullptr) {
                gfx::allocate_command_and_initialize<Upload_Model_Command>(rq, &_charter_debug_mesh->renderer_handle, &_charter_debug_mesh.value());
            }
        }

        if (_tex_test_grid.has_value()) {
            if (_charter_debug_mesh.has_value()) {
                gfx::allocate_command_and_initialize<Render_Model>(rq, _charter_debug_mesh->renderer_handle, *_tex_test_grid, _transform);
            }
        } else {
            gfx::allocate_command_and_initialize<Load_Texture_Command>(rq, &_tex_test_grid, test_grid_png, test_grid_png_len);
            if (_charter_debug_mesh.has_value()) {
                gfx::allocate_command_and_initialize<Render_Debug_Mesh<Charter_Debug_Mesh>>(rq, &_charter_debug_mesh.value());
            }
        }
        if (_debug_mesh.has_value()) {
            gfx::allocate_command_and_initialize<Render_Debug_Mesh<Debug_Mesh>>(rq, &_debug_mesh.value());
        }

        for (auto &id : _models_destroying) {
            if (id != nullptr) {
                gfx::allocate_command_and_initialize<Destroy_Model_Command>(rq, id);
            }
        }
        _models_destroying.clear();
    }

    void do_generate_mesh() override {
        auto mesh = marching_cubes::generate(_metaballs, _mc_params);

        _psp_mesh.position = mesh.positions;
        _psp_mesh.normal = mesh.normal;

        _psp_mesh.elements.clear();
        std::transform(mesh.indices.begin(), mesh.indices.end(), std::back_inserter(_psp_mesh.elements), [&](unsigned idx) { return (size_t)idx; });

        if (_charter_debug_mesh) {
            _models_destroying.push_back(_charter_debug_mesh->renderer_handle);
        }
        if (_debug_mesh) {
            _models_destroying.push_back(_debug_mesh->renderer_handle);
        }

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

        if (_charter_debug_mesh) {
            _models_destroying.push_back(_charter_debug_mesh->renderer_handle);
        }
        if (_debug_mesh) {
            _models_destroying.push_back(_debug_mesh->renderer_handle);
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

    std::optional<gfx::Texture_ID> _tex_test_grid;
    gfx::Transform _transform{ glm::vec3(), glm::quat(), glm::vec3(1, 1, 1) };

    // List of models to be destroyed
    std::vector<gfx::Model_ID> _models_destroying;
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
