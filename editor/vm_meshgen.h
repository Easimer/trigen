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

struct Basic_Mesh {
    gfx::Model_ID renderer_handle = nullptr;

    std::vector<std::array<float, 3>> positions;
    std::vector<std::array<float, 3>> normals;
    std::vector<unsigned> elements;
};

struct Unwrapped_Mesh : public Basic_Mesh {
    std::vector<glm::vec2> uv;
};

struct Input_Texture {
    std::unique_ptr<uint8_t[]> data;
    Trigen_Texture info;
};

class VM_Meshgen : public QObject {
    Q_OBJECT;

public:
    VM_Meshgen(QWorld const *world, Entity_Handle ent);

    bool checkEntity() const;
    void onRender(gfx::Render_Queue *rq);
    void foreachInputTexture(std::function<void(Trigen_Texture_Kind, char const *, Input_Texture &)> const &callback);

public slots:
    void numberOfSubdivionsChanged(int subdivisions);
    void metaballRadiusChanged(float metaballRadius);
    void loadTextureFromPath(Trigen_Texture_Kind kind, char const *path);
    void resolutionChanged(int resolution);

    /**
     * Call this from the view when the user wants to export the generated
     * mesh.
     */
    void onExportClicked();

    /**
     * Call this from slot that received the showExportFileDialog
     * signal when the user has entered a non-empty export path.
     *
     * @param path File path entered by the user.
     */
    void onExportPathAvailable(QString const &path);

    void renderNormalsOptionChanged(bool renderNormals) { _renderNormals = renderNormals; }

signals:
    /**
     * Called when the export process succeeded without any errors
     * and the file was saved to the disk.
     */
    void exported();

    /**
     * Emitted if anything goes wrong during export.
     * May be used by the view to display an error message box.
     *
     * @param msg Error message
     */
    void exportError(QString const &msg);

    /**
     * Emitted after onExportClicked is called and the mesh is ready
     * for export.
     * The receiver slot should call onExportPathAvailable when the
     * user has entered a non-empty export path.
     */
    void showExportFileDialog();

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

    std::optional<Unwrapped_Mesh> _unwrappedMesh;

    PSP::Material _inputMaterial;
    Input_Texture _texBase;
    Input_Texture _texNormal;
    Input_Texture _texHeight;
    Input_Texture _texRoughness;
    Input_Texture _texAo;

    Trigen_Texture _texOutBase;
    Trigen_Texture _texOutNormal;
    Trigen_Texture _texOutHeight;
    Trigen_Texture _texOutRoughness;
    Trigen_Texture _texOutAo;

    gfx::Texture_ID _texOutBaseHandle = nullptr;
    gfx::Texture_ID _texOutNormalHandle = nullptr;

    std::vector<gfx::Model_ID> _modelsDestroying;
    std::vector<gfx::Texture_ID> _texturesDestroying;

    bool _renderNormals = false;
};
