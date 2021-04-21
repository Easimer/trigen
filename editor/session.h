// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: session declaration
//

#pragma once

#include <memory>
#include <optional>
#include <arcball_camera.h>
#include <QObject>

#include "world_qt.h"

#include <softbody.h>
#include <r_queue.h>
#include <r_cmd/softbody.h>

enum class Session_Gizmo_Mode {
    Translation,
    Rotation,
    Scaling,
};

class Session : public QObject {
    Q_OBJECT;
public:
    Session(char const *name);

    std::string name() const { return _name; }
    void createPlant(sb::Config const &cfg);
    bool isRunning() const { return _isRunning; }
	void addColliderFromPath(char const *path);
    QWorld const *world() const { return &_world; }
    void selectEntity(int index);
    void deselectEntity();

public slots:
    void onTick(float deltaTime);
    void onRender(gfx::Render_Queue *rq);
    void setRunning(bool isRunning);
    void onMeshUpload(gfx::Render_Queue *rq);
    void setGizmoMode(Session_Gizmo_Mode mode);

private:
    std::string _name;
    Softbody_Render_Parameters _renderParams;
    QWorld _world;
    bool _isRunning = false;
    std::optional<Entity_Handle> _selectedEntity;

    std::vector<Entity_Handle> _pendingColliderMeshUploads;

    glm::mat4 _matView;
    glm::mat4 _matProj;
    Session_Gizmo_Mode _gizmoMode = Session_Gizmo_Mode::Translation;
};