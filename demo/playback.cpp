// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#include "playback.h"
#include <algorithm>
#include <iterator>

struct Grow_Work_Info {
    Playback *playback;
    float dt;
};

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
    if (_visTree.renderable) {
        rq->Submit(_visTree.renderable, transform);
    }
    if (_visFoliage.renderable) {
        rq->Submit(_visFoliage.renderable, transform);
    }
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
Playback::beginRegenerateRenderable() {
    _workMeshRegen.data = this;
    uv_queue_work(
        _app->Loop(), &_workMeshRegen, &Playback::regenerateRenderable,
        &Playback::regenerateRenderableAfter);
}

void
Playback::regenerateRenderable() {
    auto *sim = _app->Simulation();
    Trigen_Metaballs_SetScale(sim, 1.0f);
    Trigen_Metaballs_Regenerate(sim);
    Trigen_Mesh_Regenerate(sim);
    Trigen_Foliage_Parameters params;
    params.kind = Trigen_FoliageParam_Scale;
    params.valuef32 = 0.5f;
    Trigen_Foliage_SetParameters(sim, &params);
    Trigen_Foliage_Regenerate(sim);

    Trigen_Painting_SetOutputResolution(sim, 2048, 2048);
    Trigen_Painting_Regenerate(sim);

    Trigen_Mesh_GetMesh(sim, &_mesh);
    Trigen_Foliage_GetMesh(sim, &_meshFoliage);
}

void
Playback::regenerateRenderableAfter() {
    auto *renderer = _app->Renderer();
    auto *sim = _app->Simulation();

    renderer->BeginModelManagement();
    auto model = UploadTrigenMeshToTopo(renderer, _mesh);
    auto modelFoliage = UploadTrigenMeshToTopo(renderer, _meshFoliage);
    renderer->FinishModelManagement();
    Trigen_Mesh_FreeMesh(&_mesh);
    Trigen_Foliage_FreeMesh(&_meshFoliage);
    _mesh = {};
    _meshFoliage = {};

    Trigen_Texture textureDiffuse, textureNormal;

    Trigen_Painting_GetOutputTexture(sim, Trigen_Texture_BaseColor, &textureDiffuse);
    Trigen_Painting_GetOutputTexture(sim, Trigen_Texture_NormalMap, &textureNormal);

    unsigned char imageGreen[4] = {0, 255, 0, 255};

    topo::Texture_ID texDiffuse, texNormal, texLeaves;
    topo::Material_ID matTree, matFoliage;
    topo::Renderable_ID renderableTree, renderableFoliage;
    renderer->CreateTexture(
        &texDiffuse, textureDiffuse.width, textureDiffuse.height,
        topo::Texture_Format::SRGB888, textureDiffuse.image);
    renderer->CreateTexture(
        &texNormal, textureNormal.width, textureNormal.height,
        topo::Texture_Format::RGB888, textureNormal.image);
    if (_texLeaf != nullptr) {
        texLeaves = _texLeaf;
    } else {
        printf("[Playback] leaf texture was nullptr, what's going on?\n");
        renderer->CreateTexture(
            &texLeaves, 1, 1, topo::Texture_Format::RGBA8888, imageGreen);
    }
    renderer->CreateLitMaterial(&matTree, texDiffuse, texNormal);
    renderer->CreateUnlitTransparentMaterial(&matFoliage, texLeaves);
    renderer->CreateRenderable(&renderableTree, model, matTree);
    renderer->CreateRenderable(&renderableFoliage, modelFoliage, matFoliage);
    
    if (_visFoliage.texDiffuse == _texLeaf) {
        // Save the leaf texture from being destroyed by the clear() below
        _visFoliage.texDiffuse = nullptr;
    }

    _visTree.clear();
    _visFoliage.clear();

    _visTree
        = { renderer, texDiffuse, texNormal, matTree, model, renderableTree };
    _visFoliage = { renderer,  texLeaves,    nullptr,
                    matFoliage, modelFoliage, renderableFoliage };

    _generatingVisuals = false;
    _app->OnTreeVisualsReady();
}

void
Playback::oneshotGrow(float dt) {
    auto *sim = _app->Simulation();
    Trigen_Grow(sim, dt);
}

void
Playback::oneshotGrowAfter() {
    _simulationLocked = false;
    _app->OnSimulationStepOver();
}

bool
Playback::doOneshot(float dt) {
    if (!_visTree && !_generatingVisuals && !_growing) {
        printf("[Playback] Begin growing\n");
        _growing = true;
        _currentTime = 0;
    }

    if (_growing) {
        if (_currentTime < _demo.oneshot.at) {
            if (!_simulationLocked) {
                auto *workGrow = new uv_work_t;
                workGrow->data = new Grow_Work_Info({ this, dt });
                _simulationLocked = true;
                uv_queue_work(
                    _app->Loop(), workGrow, &Playback::oneshotGrow,
                    &Playback::oneshotGrowAfter);
            } else {
                _currentTime -= dt;
            }
        } else {
            printf("[Playback] Begin regenerate renderable\n");
            _growing = false;
            _generatingVisuals = true;
            beginRegenerateRenderable();
        }
    }

    if (!_visTree && _generatingVisuals) {
        _currentTime = 0;
    }

    if (_visTree) {
        _generatingVisuals = false;
        if (_currentTime >= _demo.oneshot.hold) {
            return true;
        }
    }

    return false;
}

bool
Playback::doTimelapse(float dt) {
    return false;
}

void
Playback::regenerateRenderable(uv_work_t *work) {
    auto playback = (Playback *)work->data;
    playback->regenerateRenderable();
}

void
Playback::regenerateRenderableAfter(uv_work_t *work, int status) {
    auto playback = (Playback *)work->data;
    playback->regenerateRenderableAfter();
}

void
Playback::oneshotGrow(uv_work_t *work) {
    auto *info = (Grow_Work_Info *)work->data;
    info->playback->oneshotGrow(info->dt);
}

void
Playback::oneshotGrowAfter(uv_work_t *work, int status) {
    auto *info = (Grow_Work_Info *)work->data;
    info->playback->oneshotGrowAfter();
    delete info;
    delete work;
}

