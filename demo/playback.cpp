// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#include "playback.h"
#include <algorithm>
#include <iterator>

bool
Playback::step(float dt) {
    _currentTime += dt;

    switch (_demo.kind) {
    case Demo::ONESHOT:
        return doOneshot(dt);
    case Demo::TIMELAPSE:
        return doTimelapse(dt);
    }

    return false;
}

void
Playback::render(topo::IRender_Queue *rq) {
    topo::Transform transform;
    rq->Submit(_visTree.renderable, transform);
    rq->Submit(_visFoliage.renderable, transform);
}

static topo::Model_ID
UploadTrigenMeshToTopo(topo::IInstance* renderer, Trigen_Mesh const& mesh) {
    topo::Model_Descriptor modelDesc;
    std::vector<unsigned> elements;
    std::transform(
        mesh.indices, mesh.indices + mesh.triangle_count * 3,
        std::back_inserter(elements), [](tg_u64 x) { return (unsigned)x; });
    modelDesc.elements = elements.data();
    modelDesc.element_count = mesh.triangle_count * 3;
    modelDesc.vertices = mesh.positions;
    modelDesc.normals = mesh.normals;
    modelDesc.vertex_count = mesh.position_count;
    modelDesc.uv = mesh.uvs;

    topo::Model_ID ret;
    renderer->CreateModel(&ret, &modelDesc);
    return ret;
}

void
Playback::regenerateRenderable() {
    _visTree.clear();
    _visFoliage.clear();
    Trigen_Mesh_SetSubdivisions(_simulation, 64);
    Trigen_Metaballs_SetScale(_simulation, 0.1);
    Trigen_Metaballs_Regenerate(_simulation);
    Trigen_Mesh_Regenerate(_simulation);
    Trigen_Foliage_Regenerate(_simulation);

    Trigen_Mesh mesh, meshFoliage;
    Trigen_Mesh_GetMesh(_simulation, &mesh);
    Trigen_Foliage_GetMesh(_simulation, &meshFoliage);

    _renderer->BeginModelManagement();
    auto model = UploadTrigenMeshToTopo(_renderer, mesh);
    auto modelFoliage = UploadTrigenMeshToTopo(_renderer, meshFoliage);
    _renderer->FinishModelManagement();
    Trigen_Mesh_FreeMesh(&mesh);

    Trigen_Painting_SetOutputResolution(_simulation, 512, 512);
    Trigen_Painting_Regenerate(_simulation);

    Trigen_Texture texture;
    topo::Texture_ID texDiffuse;
    topo::Texture_ID texNormal;
    topo::Texture_ID texLeaves;

    Trigen_Painting_GetOutputTexture(
        _simulation, Trigen_Texture_BaseColor, &texture);
    _renderer->CreateTexture(
        &texDiffuse, texture.width, texture.height,
        topo::Texture_Format::SRGB888, texture.image);
    Trigen_Painting_GetOutputTexture(
        _simulation, Trigen_Texture_NormalMap, &texture);
    _renderer->CreateTexture(
        &texNormal, texture.width, texture.height,
        topo::Texture_Format::RGB888, texture.image);

    unsigned char imageGreen[4] = {0, 255, 0, 255};
    _renderer->CreateTexture(
        &texLeaves, 1, 1, topo::Texture_Format::RGBA8888, imageGreen);

    topo::Material_ID matTree, matFoliage;
    _renderer->CreateLitMaterial(&matTree, texDiffuse, texNormal);
    _renderer->CreateUnlitTransparentMaterial(&matFoliage, texLeaves);
    topo::Renderable_ID renderableTree, renderableFoliage;
    _renderer->CreateRenderable(&renderableTree, model, matTree);
    _renderer->CreateRenderable(&renderableFoliage, modelFoliage, matFoliage);
    _visTree
        = { _renderer, texDiffuse, texNormal, matTree, model, renderableTree };
    _visFoliage = { _renderer,  texLeaves,    nullptr,
                    matFoliage, modelFoliage, renderableFoliage };
}

bool
Playback::doOneshot(float dt) {
    if (_currentTime >= _demo.oneshot.hold) {
        return true;
    }
    if (!_visTree) {
        Trigen_Grow(_simulation, _demo.oneshot.at);
        regenerateRenderable();
    }
    return false;
}

bool
Playback::doTimelapse(float dt) {
    return false;
}
