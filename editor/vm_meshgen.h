// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <QObject>

#include <array>
#include <cstdint>
#include <memory>
#include <vector>
#include <functional>

#include "world_qt.h"

#include <marching_cubes.h>
#include <psp/psp.h>
#include <r_queue.h>
#include <r_renderer.h>

enum class Texture_Kind {
    Base,
    Normal,
    Height,
    Roughness,
    AO,
};

struct Basic_Mesh {
    gfx::Model_ID renderer_handle = nullptr;

    std::vector<std::array<float, 3>> positions;
    std::vector<unsigned> elements;
};

struct Unwrapped_Mesh : public Basic_Mesh {
    std::vector<glm::vec2> uv;
};

struct Input_Texture {
    std::unique_ptr<uint8_t[]> data;
    PSP::Texture info;
};

class VM_Meshgen : public QObject {
    Q_OBJECT;

public:
    VM_Meshgen(QWorld const *world, Entity_Handle ent);

    bool checkEntity() const;
    void onRender(gfx::Render_Queue *rq);
    void foreachInputTexture(std::function<void(Texture_Kind, char const *, Input_Texture &)> const &callback);

public slots:
    void numberOfSubdivionsChanged(int subdivisions);
    void metaballRadiusChanged(float metaballRadius);
    void loadTextureFromPath(Texture_Kind kind, char const *path);
    void resolutionChanged(int resolution);

protected:
    void regenerateMetaballs();
    void regenerateMesh();
    void regenerateUVs();
    void repaintMesh();

    void destroyModel(gfx::Model_ID handle);
    void cleanupModels(gfx::Render_Queue *rq);

private:
    QWorld const *_world;
    Entity_Handle _ent;

    std::vector<marching_cubes::metaball> _metaballs;

    marching_cubes::params _meshgenParams;
    float _metaballRadius;
    PSP::Parameters _paintParams;
    std::optional<PSP::Mesh> _pspMesh;
    std::optional<Basic_Mesh> _basicMesh;
    std::optional<Unwrapped_Mesh> _unwrappedMesh;

    PSP::Material _inputMaterial;
    Input_Texture _texBase;
    Input_Texture _texNormal;
    Input_Texture _texHeight;
    Input_Texture _texRoughness;
    Input_Texture _texAo;

    PSP::Material _outputMaterial;
    gfx::Texture_ID _texOutBase = nullptr;

    std::unique_ptr<PSP::IPainter> _painter;

    std::vector<gfx::Model_ID> _modelsDestroying;
    std::vector<gfx::Texture_ID> _texturesDestroying;
};
