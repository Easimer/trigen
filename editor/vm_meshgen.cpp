// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"
#include "vm_meshgen.h"

#include <set>

#include <mesh_export.h>
#include <r_cmd/general.h>

#include <stb_image.h>

/**
 * @param T Either Basic_Mesh or Unwrapped_Mesh
 */
template<typename T>
class Upload_Model_Command : public gfx::IRender_Command {
public:
    Upload_Model_Command(gfx::Model_ID *out_id, T const *mesh) : _out_id(out_id), _mesh(mesh) {
        assert(_out_id != nullptr);
        assert(_mesh != nullptr);
    }

    void execute(gfx::IRenderer *renderer) override {
        gfx::Model_Descriptor model{};

        assert(*_out_id == nullptr);
        fill(&model, _mesh);

        renderer->create_model(_out_id, &model);
    }

    void fill(gfx::Model_Descriptor *d, Unwrapped_Mesh const *mesh) {
        fill(d, (Basic_Mesh *)mesh);
        d->uv = (std::array<float, 2>*)_mesh->uv.data();
    }

    void fill(gfx::Model_Descriptor *d, Basic_Mesh const *mesh) {
        d->vertex_count = _mesh->positions.size();
        d->vertices = _mesh->positions.data();
        d->element_count = _mesh->elements.size();
        d->elements = _mesh->elements.data();
    }

private:
    gfx::Model_ID *_out_id;
    T const *_mesh;
};

static Basic_Mesh convertMesh(marching_cubes::mesh const &mesh) {
    Basic_Mesh ret;

    std::transform(
        mesh.positions.begin(), mesh.positions.end(),
        std::back_inserter(ret.positions),
        [&](auto p) -> std::array<float, 3> {
            return { p.x, p.y, p.z };
        });
    ret.elements = mesh.indices;
    return ret;
}

Unwrapped_Mesh convertMesh(PSP::Mesh const &mesh) {
    Unwrapped_Mesh ret;

    std::transform(
        mesh.position.begin(), mesh.position.end(),
        std::back_inserter(ret.positions),
        [&](auto p) -> std::array<float, 3> {
            return { p.x, p.y, p.z };
        });
    std::transform(
        mesh.elements.begin(), mesh.elements.end(),
        std::back_inserter(ret.elements),
        [&](auto p) { return (unsigned)p; });

    ret.uv = mesh.uv;

    return ret;
}

VM_Meshgen::VM_Meshgen(QWorld const *world, Entity_Handle ent)
    : _world(world)
    , _ent(ent) {
    _meshgenParams.subdivisions = 2;
    _metaballRadius = 1;
}

bool VM_Meshgen::checkEntity() const {
    return _world->exists(_ent) && (_world->getMapForComponent<Plant_Component>().count(_ent) > 0);
}

void VM_Meshgen::onRender(gfx::Render_Queue *rq) {
    gfx::Transform renderTransform {
        { 0, 0, 0 },
        { 1, 0, 0, 0 },
        { 1, 1, 1 }
    };
    auto &transforms = _world->getMapForComponent<Transform_Component>();
    if (transforms.count(_ent)) {
        auto &transform = transforms.at(_ent);
        renderTransform.position = transform.position;
        renderTransform.rotation = transform.rotation;
        renderTransform.scale = transform.scale;
    }

    if (_outputMaterial.base.buffer != nullptr && _texOutBase == nullptr) {
        gfx::allocate_command_and_initialize<Upload_Texture_Command>(rq, &_texOutBase, _outputMaterial.base.buffer, _outputMaterial.base.width, _outputMaterial.base.height, gfx::Texture_Format::RGB888);
    }

    if (_unwrappedMesh.has_value()) {
        if (_unwrappedMesh->renderer_handle != nullptr) {
            // Render mesh
            if (_texOutBase != nullptr) {
                gfx::allocate_command_and_initialize<Render_Model>(rq, _unwrappedMesh->renderer_handle, _texOutBase, renderTransform);
            } else {
                gfx::allocate_command_and_initialize<Render_Untextured_Model>(rq, _unwrappedMesh->renderer_handle, renderTransform);
            }
        } else {
            gfx::allocate_command_and_initialize<Upload_Model_Command<Unwrapped_Mesh>>(rq, &_unwrappedMesh->renderer_handle, &*_unwrappedMesh);
        }
    } else if (_basicMesh.has_value()) {
        if (_basicMesh->renderer_handle != nullptr) {
            // Render basic mesh
            gfx::allocate_command_and_initialize<Render_Untextured_Model>(rq, _basicMesh->renderer_handle, renderTransform);
        } else {
            gfx::allocate_command_and_initialize<Upload_Model_Command<Basic_Mesh>>(rq, &_basicMesh->renderer_handle, &*_basicMesh);
        }
    }
}

void VM_Meshgen::foreachInputTexture(std::function<void(Texture_Kind, char const *, Input_Texture &)> const &callback) {
    callback(Texture_Kind::Base, "Base color", _texBase);
    callback(Texture_Kind::Normal, "Normal map", _texNormal);
    callback(Texture_Kind::Height, "Height map", _texHeight);
    callback(Texture_Kind::Roughness, "Roughness map", _texRoughness);
    callback(Texture_Kind::AO, "AO", _texAo);
}

void VM_Meshgen::destroyModel(gfx::Model_ID handle) {
    _modelsDestroying.push_back(handle);
}

void VM_Meshgen::cleanupModels(gfx::Render_Queue *rq) {
    for (auto handle : _modelsDestroying) {
        if (handle != nullptr) {
            gfx::allocate_command_and_initialize<Destroy_Model_Command>(rq, handle);
        }
    }

    _modelsDestroying.clear();
}

void VM_Meshgen::numberOfSubdivionsChanged(int subdivisions) {
    _meshgenParams.subdivisions = subdivisions;

    regenerateMesh();
}

void VM_Meshgen::metaballRadiusChanged(float metaballRadius) {
    _metaballRadius = metaballRadius;

    regenerateMetaballs();
}

void VM_Meshgen::loadTextureFromPath(Texture_Kind kind, char const *path) {
    int channels, width, height;
    std::unique_ptr<uint8_t[]> data;
    stbi_uc *buffer;
    if ((buffer = stbi_load(path, &width, &height, &channels, 3)) != nullptr) {
        auto size = size_t(width) * size_t(height) * 3;
        data = std::make_unique<uint8_t[]>(size);
        memcpy(data.get(), buffer, size);
        stbi_image_free(buffer);
    } else {
        assert(0);
        return;
    }

    Input_Texture *tex = nullptr;

    switch (kind) {
    case Texture_Kind::Base:
        tex = &_texBase;
        break;
    case Texture_Kind::Normal:
        tex = &_texNormal;
        break;
    case Texture_Kind::Height:
        tex = &_texHeight;
        break;
    case Texture_Kind::Roughness:
        tex = &_texRoughness;
        break;
    case Texture_Kind::AO:
        tex = &_texAo;
        break;
    }

    if (tex != nullptr) {
        tex->data = std::move(data);
        tex->info.buffer = tex->data.get();
        tex->info.width = width;
        tex->info.height = height;

        repaintMesh();
    } else {
        assert(0);
    }
}

void VM_Meshgen::resolutionChanged(int resolution) {
    _paintParams.out_width = resolution;
    _paintParams.out_height = resolution;

    repaintMesh();
}

void VM_Meshgen::onExportClicked() {
    if (!_pspMesh) {
        emit exportError("Meshgen isn't done yet!");
        return;
    }

    if (!_painter || !_painter->is_painting_done()) {
        emit exportError("Painting isn't done yet!");
        return;
    }

    emit showExportFileDialog();
}

void VM_Meshgen::onExportPathAvailable(QString const &path) {
    assert(!path.isEmpty());
    if (path.isEmpty()) {
        return;
    }

    assert(_pspMesh.has_value());
    if (!_pspMesh.has_value()) {
        emit exportError("Generated mesh has disappeared between validation and saving. Programmer error?");
    }

    auto const pathu8 = path.toUtf8();

    if (fbx_try_save(pathu8.constData(), &_pspMesh.value(), &_outputMaterial)) {
        emit exported();
    } else {
        emit exportError("Couldn't save FBX file!");
    }
}

void VM_Meshgen::regenerateMetaballs() {
    assert(checkEntity());

    _metaballs.clear();

    if (!checkEntity()) {
        return;
    }

    auto &simulation = _world->getMapForComponent<Plant_Component>().at(_ent)._sim;

    // Gather particles and connections
    std::unordered_map<sb::index_t, sb::Particle> particles;
    std::set<std::pair<sb::index_t, sb::index_t>> connections;

    for (auto iter = simulation->get_particles(); !iter->ended(); iter->step()) {
        auto p = iter->get();
        particles[p.id] = p;
    }

    for (auto iter = simulation->get_connections(); !iter->ended(); iter->step()) {
        auto c = iter->get();
        if (c.parent < c.child) {
            connections.insert({ c.parent, c.child });
        } else {
            connections.insert({ c.child, c.parent });
        }
    }

    // Generate metaballs
    for (auto &conn : connections) {
        auto p0 = particles[conn.first].position;
        auto p1 = particles[conn.second].position;
        auto s0 = particles[conn.first].size;
        auto s1 = particles[conn.second].size;
        auto dir = p1 - p0;
        auto dirLen = length(p1 - p0);
        auto sizDir = s1 - s0;
        auto steps = int((dirLen + 1) * 16.0f);

        for (int s = 0; s < steps; s++) {
            auto t = s / (float)steps;
            auto p = p0 + t * dir;
            auto size = s0 + t * sizDir;
            float radius = 8.0f;
            for (int i = 0; i < 3; i++) {
                radius = glm::max(size[i] / 2, radius);
            }
            _metaballs.push_back({ p, radius / 8 });
        }
    }

    regenerateMesh();
}

void VM_Meshgen::regenerateMesh() {
    auto mesh = marching_cubes::generate(_metaballs, _meshgenParams);

    _basicMesh = convertMesh(mesh);

    PSP::Mesh pspMesh;
    pspMesh.position = std::move(mesh.positions);
    pspMesh.normal = std::move(mesh.normal);
    std::transform(mesh.indices.begin(), mesh.indices.end(), std::back_inserter(pspMesh.elements), [&](unsigned idx) { return (size_t)idx; });

    _pspMesh.emplace(std::move(pspMesh));

    regenerateUVs();
}

void VM_Meshgen::regenerateUVs() {
    if (!_pspMesh.has_value()) {
        return;
    }

    _pspMesh->uv.clear();
    PSP::unwrap(_pspMesh.value());

    _unwrappedMesh = convertMesh(*_pspMesh);

    repaintMesh();
}

void VM_Meshgen::repaintMesh() {
    if (!_pspMesh.has_value()) {
        return;
    }

    PSP::Texture texBlack = {};
    char const blackPixel[3] = { 0, 0, 0 };

    texBlack.buffer = blackPixel;
    texBlack.width = 1;
    texBlack.height = 1;

    auto putPlaceholderTextureIfEmpty = [&](Input_Texture const &input, PSP::Texture &target) {
        if (input.data == nullptr) {
            target = texBlack;
        } else {
            target = input.info;
        }
    };

    assert(_pspMesh->uv.size() == _pspMesh->elements.size());

    putPlaceholderTextureIfEmpty(_texBase, _inputMaterial.base);
    putPlaceholderTextureIfEmpty(_texNormal, _inputMaterial.normal);
    putPlaceholderTextureIfEmpty(_texHeight, _inputMaterial.height);
    putPlaceholderTextureIfEmpty(_texRoughness, _inputMaterial.roughness);
    putPlaceholderTextureIfEmpty(_texAo, _inputMaterial.ao);

    _paintParams.material = &_inputMaterial;
    _paintParams.mesh = &_pspMesh.value();

    if (_texOutBase != nullptr) {
        _texturesDestroying.push_back(_texOutBase);
    }
    _texOutBase = nullptr;
    _outputMaterial = {};
    _painter = PSP::make_painter(_paintParams);

    _painter->step_painting(0);

    if (_painter->is_painting_done()) {
        _painter->result(&_outputMaterial);
    }
}

