// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <QObject>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "trigen_worker.h"
#include "world_qt.h"

struct Basic_Mesh {
    topo::Model_ID renderer_handle = nullptr;

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

class IMeshgen_Statusbar {
public:
    virtual ~IMeshgen_Statusbar() = default;

    virtual void
    setMessage(char const *message)
        = 0;

    virtual void
    setBusy(bool isBusy)
        = 0;
};

enum class Meshgen_Texture_Kind {
    BaseColor = Trigen_Texture_BaseColor,
    NormalMap = Trigen_Texture_NormalMap,
    HeightMap = Trigen_Texture_HeightMap,
    RoughnessMap = Trigen_Texture_RoughnessMap,
    AmbientOcclusionMap = Trigen_Texture_AmbientOcclusionMap,
    LeafBaseColor,
};

class VM_Meshgen : public QObject {
    Q_OBJECT;

public:
    VM_Meshgen(QWorld const *world, Entity_Handle ent, IMeshgen_Statusbar *statusBar);

    bool checkEntity() const;
    void onRender(topo::IRender_Queue *rq);
    void foreachInputTexture(std::function<void(Meshgen_Texture_Kind, char const *, Input_Texture &)> const &callback);

public slots:
    void numberOfSubdivionsChanged(int subdivisions);
    void metaballRadiusChanged(float metaballRadius);
    void loadTextureFromPath(Meshgen_Texture_Kind kind, char const *path);
    void resolutionChanged(int resolution);
    void inspectUV();

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
    void
    regenerateFoliage();
    void repaintMesh();

    void destroyModel(topo::Model_ID handle);
    void cleanupModels(topo::IInstance *rq);

protected slots:
    void onStageDone(Stage_Tag stage, Trigen_Status res, Trigen_Session session);

private:
    QWorld const *_world;
    Entity_Handle _ent;

    std::optional<Unwrapped_Mesh> _unwrappedMesh;
    std::optional<Unwrapped_Mesh> _foliageMesh;

    Input_Texture _texBase;
    Input_Texture _texNormal;
    Input_Texture _texHeight;
    Input_Texture _texRoughness;
    Input_Texture _texAo;
    Input_Texture _texLeaves;

    Trigen_Texture _texOutBase;
    Trigen_Texture _texOutNormal;
    Trigen_Texture _texOutHeight;
    Trigen_Texture _texOutRoughness;
    Trigen_Texture _texOutAo;

    topo::Texture_ID _texOutBaseHandle = nullptr;
    topo::Texture_ID _texOutNormalHandle = nullptr;
    topo::Texture_ID _texLeavesHandle = nullptr;

    std::vector<topo::Model_ID> _modelsDestroying;
    std::vector<topo::Texture_ID> _texturesDestroying;

    bool _renderNormals = false;

    Trigen_Controller _controller;
    IMeshgen_Statusbar *_statusBar;
};
