// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"
#include "vm_meshgen.h"

#include <set>

#include <r_cmd/general.h>

VM_Meshgen::VM_Meshgen(World *world, Entity_Handle ent)
    : _world(world)
    , _ent(ent) {
}

bool VM_Meshgen::checkEntity() const {
    return _world->exists(_ent) && (_world->getMapForComponent<Plant_Component>().count(_ent) > 0);
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
}

void VM_Meshgen::metaballRadiusChanged(float metaballRadius) {
    _metaballRadius = metaballRadius;
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

    PSP::Mesh pspMesh;
    pspMesh.position = std::move(mesh.positions);
    pspMesh.normal = std::move(mesh.normal);
    std::transform(mesh.indices.begin(), mesh.indices.end(), std::back_inserter(pspMesh.elements), [&](unsigned idx) { return (size_t)idx; });

    _pspMesh.emplace(std::move(pspMesh));

    regenerateUVs();
}

void VM_Meshgen::regenerateUVs() {
    assert(_pspMesh.has_value());

    if (!_pspMesh.has_value()) {
        return;
    }

    _pspMesh->uv.clear();
    PSP::unwrap(_pspMesh.value());

    repaintMesh();
}

void VM_Meshgen::repaintMesh() {
    assert(_pspMesh.has_value());

    if (!_pspMesh.has_value()) {
        return;
    }

    assert(_pspMesh->uv.size() == _pspMesh->elements.size());

    _inputMaterial.base = _texBase.info;
    _inputMaterial.normal = _texNormal.info;
    _inputMaterial.height = _texHeight.info;
    _inputMaterial.roughness = _texRoughness.info;
    _inputMaterial.ao = _texAo.info;

    _paintParams.material = &_inputMaterial;
    _paintParams.mesh = &_pspMesh.value();

    _painter = PSP::make_painter(_paintParams);
}
