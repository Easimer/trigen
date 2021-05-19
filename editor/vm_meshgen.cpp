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
        d->normals = mesh->normals.data();
    }

private:
    gfx::Model_ID *_out_id;
    T const *_mesh;
};

class Render_Normals_Command : public gfx::IRender_Command {
public:
    Render_Normals_Command(std::vector<glm::vec3> &&lines)
        : _lines(std::move(lines)) {
    }

    void execute(gfx::IRenderer *renderer) {
        renderer->draw_lines(_lines.data(), _lines.size() / 2, glm::vec3(0, 0, 0), glm::vec3(0.4, 0.4, 0.8), glm::vec3(0.4, 0.4, 1.0));
    }

private:
    std::vector<glm::vec3> _lines;
};

static Basic_Mesh convertMesh(marching_cubes::mesh const &mesh) {
    Basic_Mesh ret;

    std::transform(
        mesh.positions.begin(), mesh.positions.end(),
        std::back_inserter(ret.positions),
        [&](auto p) -> std::array<float, 3> {
            return { p.x, p.y, p.z };
        });
    std::transform(
        mesh.normal.begin(), mesh.normal.end(),
        std::back_inserter(ret.normals),
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
        mesh.normal.begin(), mesh.normal.end(),
        std::back_inserter(ret.normals),
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

Unwrapped_Mesh convertMesh(Trigen_Mesh const &mesh) {
    Unwrapped_Mesh ret;

    std::transform(
        (glm::vec3 *)mesh.positions, (glm::vec3 *)(mesh.positions + mesh.position_count * 3),
        std::back_inserter(ret.positions),
        [&](auto p) -> std::array<float, 3> {
            return { p.x, p.y, p.z };
        });
    std::transform(
        (glm::vec3 *)mesh.normals, (glm::vec3 *)(mesh.normals + mesh.normal_count),
        std::back_inserter(ret.normals),
        [&](auto p) -> std::array<float, 3> {
            return { p.x, p.y, p.z };
        });
    std::transform(
        (size_t *)mesh.vertex_indices, (size_t *)(mesh.vertex_indices + mesh.triangle_count * 3),
        std::back_inserter(ret.elements),
        [&](auto p) { return (unsigned)p; });

    std::transform(
        (glm::vec2*)mesh.uvs, (glm::vec2*)(mesh.uvs + mesh.triangle_count * 3 * 2),
        std::back_inserter(ret.uv),
        [&](auto p) -> glm::vec2 {
            return { p.x, p.y };
        });

    return ret;
}

VM_Meshgen::VM_Meshgen(QWorld const *world, Entity_Handle ent)
    : _world(world)
    , _ent(ent) {
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

    if (_texOutBase.image != nullptr && _texOutBaseHandle == nullptr) {
        gfx::allocate_command_and_initialize<Upload_Texture_Command>(rq, &_texOutBaseHandle, _texOutBase.image, _texOutBase.width, _texOutBase.height, gfx::Texture_Format::RGB888);
    }

    if (_texOutNormal.image != nullptr && _texOutNormalHandle == nullptr) {
        gfx::allocate_command_and_initialize<Upload_Texture_Command>(rq, &_texOutNormalHandle, _texOutNormal.image, _texOutNormal.width, _texOutNormal.height, gfx::Texture_Format::RGB888);
    }

    if (_unwrappedMesh.has_value()) {
        if (_unwrappedMesh->renderer_handle != nullptr) {
            // Render mesh
            if (_texOutBaseHandle != nullptr && _texOutNormalHandle != nullptr) {
                // gfx::allocate_command_and_initialize<Render_Model>(rq, _unwrappedMesh->renderer_handle, _texOutBaseHandle, _texOutNormalHandle, renderTransform);
                gfx::allocate_command_and_initialize<Render_Model>(rq, _unwrappedMesh->renderer_handle, _texOutBaseHandle, renderTransform);
            } else {
                gfx::allocate_command_and_initialize<Render_Untextured_Model>(rq, _unwrappedMesh->renderer_handle, renderTransform);
            }
        } else {
            gfx::allocate_command_and_initialize<Upload_Model_Command<Unwrapped_Mesh>>(rq, &_unwrappedMesh->renderer_handle, &*_unwrappedMesh);
        }

    }

    if (_unwrappedMesh.has_value()) {
        if (_renderNormals) {
            std::vector<glm::vec3> lines;
            // TODO: for each vertex, draw the normal
            gfx::allocate_command_and_initialize<Render_Normals_Command>(rq, std::move(lines));
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
    if (!checkEntity()) {
        return;
    }

    auto session = _world->getMapForComponent<Plant_Component>().at(_ent).session;
    Trigen_Mesh_SetSubdivisions(*session, subdivisions);

    regenerateMesh();
}

void VM_Meshgen::metaballRadiusChanged(float metaballRadius) {
    if (!checkEntity()) {
        return;
    }

    auto session = _world->getMapForComponent<Plant_Component>().at(_ent).session;
    Trigen_Metaballs_SetRadius(*session, metaballRadius);

    regenerateMetaballs();
}

static Trigen_Texture_Kind MapTextureKindToTrigenTextureKind(Texture_Kind kind) {
    switch (kind) {
    case Texture_Kind::Base:
        return Trigen_Texture_BaseColor;
    case Texture_Kind::Normal:
        return Trigen_Texture_NormalMap;
    case Texture_Kind::Height:
        return Trigen_Texture_HeightMap;
    case Texture_Kind::Roughness:
        return Trigen_Texture_RoughnessMap;
    case Texture_Kind::AO:
        return Trigen_Texture_AmbientOcclusionMap;
    }

    std::abort();
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
        tex->info.image = tex->data.get();
        tex->info.width = width;
        tex->info.height = height;

        auto session = _world->getMapForComponent<Plant_Component>().at(_ent).session;
        Trigen_Painting_SetInputTexture(session->handle(), MapTextureKindToTrigenTextureKind(kind), &tex->info);

        repaintMesh();
    } else {
        assert(0);
    }
}

void VM_Meshgen::resolutionChanged(int resolution) {
    auto session = _world->getMapForComponent<Plant_Component>().at(_ent).session;
    Trigen_Painting_SetOutputResolution(session->handle(), resolution, resolution);

    repaintMesh();
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

    PSP::Material outputMaterial;

    auto convertTrigenTextureToPSPTexture = [&](PSP::Texture &pt, Trigen_Texture const &tt) {
        pt.buffer = tt.image;
        pt.width = tt.width;
        pt.height = tt.height;
    };

    convertTrigenTextureToPSPTexture(outputMaterial.base, _texOutBase);
    convertTrigenTextureToPSPTexture(outputMaterial.normal, _texOutNormal);
    convertTrigenTextureToPSPTexture(outputMaterial.height, _texOutHeight);
    convertTrigenTextureToPSPTexture(outputMaterial.roughness, _texOutRoughness);
    convertTrigenTextureToPSPTexture(outputMaterial.ao, _texOutAo);

    /*
    if (fbx_try_save(pathu8.constData(), &_unwrappedMesh.value(), &outputMaterial)) {
        emit exported();
    } else {
        emit exportError("Couldn't save FBX file!");
    }
    */
}

void VM_Meshgen::regenerateMetaballs() {
    assert(checkEntity());

    if (!checkEntity()) {
        return;
    }

    auto session = _world->getMapForComponent<Plant_Component>().at(_ent).session;
    if (Trigen_Metaballs_Regenerate(*session) == Trigen_OK) {
        regenerateMesh();
    }
}

void VM_Meshgen::regenerateMesh() {
    auto session = _world->getMapForComponent<Plant_Component>().at(_ent).session;
    
    if (Trigen_Mesh_Regenerate(*session) == Trigen_OK) {
        Trigen_Mesh mesh;
        Trigen_Mesh_GetMesh(*session, &mesh);
        _unwrappedMesh = convertMesh(mesh);
        Trigen_Mesh_FreeMesh(&mesh);

        regenerateUVs();
    }
}

void VM_Meshgen::regenerateUVs() {
    // TODO: remove me
    repaintMesh();
}

void VM_Meshgen::repaintMesh() {
    auto session = _world->getMapForComponent<Plant_Component>().at(_ent).session;
    Trigen_Painting_Regenerate(*session);

    Trigen_Painting_GetOutputTexture(*session, Trigen_Texture_BaseColor, &_texOutBase);
    Trigen_Painting_GetOutputTexture(*session, Trigen_Texture_NormalMap, &_texOutNormal);
    Trigen_Painting_GetOutputTexture(*session, Trigen_Texture_HeightMap, &_texOutHeight);
    Trigen_Painting_GetOutputTexture(*session, Trigen_Texture_RoughnessMap, &_texOutRoughness);
    Trigen_Painting_GetOutputTexture(*session, Trigen_Texture_AmbientOcclusionMap, &_texOutAo);

    _texturesDestroying.push_back(_texOutNormalHandle);
    _texOutNormalHandle = nullptr;
    _texturesDestroying.push_back(_texOutBaseHandle);
    _texOutBaseHandle = nullptr;
}

