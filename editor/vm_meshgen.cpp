// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"
#include "vm_meshgen.h"

#include <set>

#include <QtDebug>

#include "trigen_worker.h"

#include <uv_inspect.hpp>
#include <mesh_export.h>

#include <stb_image.h>

Unwrapped_Mesh convertMesh(Trigen_Mesh const &mesh) {
    Unwrapped_Mesh ret;

    std::transform(
        (glm::vec3 *)mesh.positions, (glm::vec3 *)(mesh.positions + mesh.position_count * 3),
        std::back_inserter(ret.positions),
        [&](auto p) -> std::array<float, 3> {
            return { p.x, p.y, p.z };
        });
    std::transform(
        (glm::vec3 *)mesh.normals, (glm::vec3 *)(mesh.normals + mesh.normal_count * 3),
        std::back_inserter(ret.normals),
        [&](auto p) -> std::array<float, 3> {
            return { p.x, p.y, p.z };
        });
    std::transform(
        (size_t *)mesh.indices, (size_t *)(mesh.indices + mesh.triangle_count * 3),
        std::back_inserter(ret.elements),
        [&](auto p) { return (unsigned)p; });

    std::transform(
        (glm::vec2*)mesh.uvs, (glm::vec2*)(mesh.uvs + mesh.position_count * 2),
        std::back_inserter(ret.uv),
        [&](auto p) -> glm::vec2 {
            return { p.x, p.y };
        });

    return ret;
}

VM_Meshgen::VM_Meshgen(QWorld const *world, Entity_Handle ent, IMeshgen_Statusbar *statusBar)
    : _world(world)
    , _ent(ent)
    , _texOutBase{}
    , _texOutNormal{}
    , _texOutHeight{}
    , _texOutRoughness{}
    , _texOutAo{}
    , _statusBar(statusBar) {
    connect(&_controller, &Trigen_Controller::onResult, this, &VM_Meshgen::onStageDone);
}

bool VM_Meshgen::checkEntity() const {
    return _world->exists(_ent) && (_world->getMapForComponent<Plant_Component>().count(_ent) > 0);
}

void VM_Meshgen::onRender(topo::IRender_Queue *rq) {
    topo::Transform renderTransform {
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

}

void VM_Meshgen::foreachInputTexture(std::function<void(Meshgen_Texture_Kind, char const *, Input_Texture &)> const &callback) {
    callback(Meshgen_Texture_Kind::BaseColor, "Base color", _texBase);
    callback(Meshgen_Texture_Kind::NormalMap, "Normal map", _texNormal);
    callback(Meshgen_Texture_Kind::HeightMap, "Height map", _texHeight);
    callback(Meshgen_Texture_Kind::RoughnessMap, "Roughness map", _texRoughness);
    callback(Meshgen_Texture_Kind::AmbientOcclusionMap, "AO", _texAo);
    callback(Meshgen_Texture_Kind::LeafBaseColor, "Leaves", _texLeaves);
}

void VM_Meshgen::destroyModel(topo::Model_ID handle) {
    _modelsDestroying.push_back(handle);
}

void VM_Meshgen::cleanupModels(topo::IInstance *renderer) {
    for (auto handle : _modelsDestroying) {
        if (handle != nullptr) {
        }
    }

    _modelsDestroying.clear();
}

void VM_Meshgen::numberOfSubdivionsChanged(int subdivisions) {
    if (!checkEntity()) {
        return;
    }

    auto session = _world->getMapForComponent<Plant_Component>().at(_ent).session;
    Trigen_Mesh_SetSubdivisions(session->handle(), subdivisions);

    regenerateMesh();
}

void VM_Meshgen::metaballRadiusChanged(float metaballRadius) {
    if (!checkEntity()) {
        return;
    }

    auto session = _world->getMapForComponent<Plant_Component>().at(_ent).session;
    Trigen_Metaballs_SetScale(session->handle(), metaballRadius);

    regenerateMetaballs();
}

void VM_Meshgen::loadTextureFromPath(Meshgen_Texture_Kind kind, char const *path) {
    int channels, width, height;
    std::unique_ptr<uint8_t[]> data;
    stbi_uc *buffer;
    auto numComponents = 3;

    if (kind == Meshgen_Texture_Kind::LeafBaseColor) {
        // Leaf texture is RGBA
        numComponents = 4;
    }

    if ((buffer = stbi_load(path, &width, &height, &channels, numComponents)) != nullptr) {
        auto size = size_t(width) * size_t(height) * numComponents;
        data = std::make_unique<uint8_t[]>(size);
        memcpy(data.get(), buffer, size);
        stbi_image_free(buffer);
    } else {
        assert(0);
        return;
    }

    Input_Texture *tex = nullptr;

    switch (kind) {
    case Meshgen_Texture_Kind::BaseColor:
        tex = &_texBase;
        break;
    case Meshgen_Texture_Kind::NormalMap:
        tex = &_texNormal;
        break;
    case Meshgen_Texture_Kind::HeightMap:
        tex = &_texHeight;
        break;
    case Meshgen_Texture_Kind::RoughnessMap:
        tex = &_texRoughness;
        break;
    case Meshgen_Texture_Kind::AmbientOcclusionMap:
        tex = &_texAo;
        break;
    case Meshgen_Texture_Kind::LeafBaseColor:
        tex = &_texLeaves;
        break;
    }

    if (tex != nullptr) {
        tex->data = std::move(data);
        tex->info.image = tex->data.get();
        tex->info.width = width;
        tex->info.height = height;

        if(kind < Meshgen_Texture_Kind::LeafBaseColor) {
            // The foliage is a separate mesh, so the leaf texture is not used
            // during mesh painting.
            auto trigen_kind = (Trigen_Texture_Kind)kind;
            auto session = _world->getMapForComponent<Plant_Component>().at(_ent).session;
            Trigen_Painting_SetInputTexture(session->handle(), trigen_kind, &tex->info);

            repaintMesh();
        }
    } else {
        assert(0);
    }
}

void VM_Meshgen::resolutionChanged(int resolution) {
    auto session = _world->getMapForComponent<Plant_Component>().at(_ent).session;
    Trigen_Painting_SetOutputResolution(session->handle(), resolution, resolution);

    repaintMesh();
}

void VM_Meshgen::inspectUV() {
    if (_unwrappedMesh.has_value()) {
        uv_inspector::inspect(_unwrappedMesh->uv.data(), _unwrappedMesh->elements.data(), _unwrappedMesh->elements.size());
    }
}

void VM_Meshgen::onExportClicked() {
    emit showExportFileDialog();
}

void VM_Meshgen::onExportPathAvailable(QString const &path) {
    assert(!path.isEmpty());
    if (path.isEmpty()) {
        return;
    }

    assert(_unwrappedMesh.has_value());
    if (!_unwrappedMesh.has_value()) {
        emit exportError("Generated mesh has disappeared between validation and saving. Programmer error?");
    }

    auto const pathu8 = path.toUtf8();

    auto session = _world->getMapForComponent<Plant_Component>().at(_ent).session;
    Trigen_Material outputMaterial = {
        &_texOutBase,
        &_texOutNormal,
        &_texOutHeight,
        &_texOutRoughness,
        &_texOutAo,
    };

    auto mesh = trigen::Mesh::make(session->handle());
    auto foliageMesh = trigen::Foliage_Mesh::make(session->handle());

    Export_Model model;
    model.mesh = *mesh;
    model.material = outputMaterial;
    model.foliageMesh = *foliageMesh;
    model.leafTexture = &_texLeaves.info;

    if (fbx_try_save(pathu8.constData(), model)) {
        emit exported();
    } else {
        emit exportError("Couldn't save FBX file!");
    }
}

void VM_Meshgen::regenerateMetaballs() {
    assert(checkEntity());

    if (!checkEntity()) {
        return;
    }

    if (_statusBar) {
        _statusBar->setBusy(true);
        _statusBar->setMessage("Generating plant surface...");
    }

    _controller.session = _world->getMapForComponent<Plant_Component>().at(_ent).session->handle();
    _controller.execute(Stage_Tag::Metaballs, [](Trigen_Session session) {
        return Trigen_Metaballs_Regenerate(session);
    });
}

void VM_Meshgen::regenerateMesh() {
    assert(checkEntity());
    
    if (!checkEntity()) {
        return;
    }

    if (_statusBar) {
        _statusBar->setBusy(true);
        _statusBar->setMessage("Generating plant mesh...");
    }

    _controller.session = _world->getMapForComponent<Plant_Component>().at(_ent).session->handle();
    _controller.execute(Stage_Tag::Mesh, [](Trigen_Session session) {
        return Trigen_Mesh_Regenerate(session);
    });
}

void
VM_Meshgen::regenerateFoliage() {
    assert(checkEntity());
    
    if (!checkEntity()) {
        return;
    }

    if (_statusBar) {
        _statusBar->setBusy(true);
        _statusBar->setMessage("Generating plant foliage mesh...");
    }

    _controller.session = _world->getMapForComponent<Plant_Component>().at(_ent).session->handle();
    _controller.execute(Stage_Tag::Foliage, [](Trigen_Session session) {
        return Trigen_Foliage_Regenerate(session);
    });
}

static void clear(Trigen_Texture &tex) {
    tex.image = nullptr;
    tex.width = tex.height = 0;
}

void VM_Meshgen::onStageDone(Stage_Tag stage, Trigen_Status res, Trigen_Session session) {
    switch (stage) {
        case Stage_Tag::Metaballs:
        {
            regenerateMesh();
            break;
        }
        case Stage_Tag::Mesh:
        {
            try {
                auto mesh = trigen::Mesh::make(session);
                _unwrappedMesh = convertMesh(*mesh);
                regenerateFoliage();
            } catch(trigen::Exception const &ex) {
            }
            break;
        }
        case Stage_Tag::Foliage: {
            try {
                auto mesh = trigen::Foliage_Mesh::make(session);
                _foliageMesh = convertMesh(*mesh);
                repaintMesh();
            } catch(trigen::Exception const &ex) {
            }
            break;
        }
        case Stage_Tag::Painting:
        {
            if(res == Trigen_OK) {
                Trigen_Painting_GetOutputTexture(session, Trigen_Texture_BaseColor, &_texOutBase);
                Trigen_Painting_GetOutputTexture(session, Trigen_Texture_NormalMap, &_texOutNormal);
                Trigen_Painting_GetOutputTexture(session, Trigen_Texture_HeightMap, &_texOutHeight);
                Trigen_Painting_GetOutputTexture(session, Trigen_Texture_RoughnessMap, &_texOutRoughness);
                Trigen_Painting_GetOutputTexture(session, Trigen_Texture_AmbientOcclusionMap, &_texOutAo);
            } else {
                qWarning() << "Trigen_Painting_Regenerate has failed with rc=" << res << '\n';
                clear(_texOutBase);
                clear(_texOutNormal);
                clear(_texOutHeight);
                clear(_texOutRoughness);
                clear(_texOutAo);
            }

            _texturesDestroying.push_back(_texOutNormalHandle);
            _texOutNormalHandle = nullptr;
            _texturesDestroying.push_back(_texOutBaseHandle);
            _texOutBaseHandle = nullptr;

            if (_statusBar) {
                _statusBar->setBusy(false);
            }
            break;
        }
    }
}

void VM_Meshgen::repaintMesh() {

    if (_statusBar) {
        _statusBar->setBusy(true);
        _statusBar->setMessage("Painting the plant's surface...");
    }

    _controller.session = _world->getMapForComponent<Plant_Component>().at(_ent).session->handle();
    _controller.execute(Stage_Tag::Painting, [](Trigen_Session session) {
        return Trigen_Painting_Regenerate(session);
    });
}

