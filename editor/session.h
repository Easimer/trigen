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

#include <trigen.hpp>

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
    void createPlant(Trigen_Parameters const &cfg);
    bool isRunning() const { return _isRunning; }
	void addColliderFromPath(char const *path);
    QWorld const *world() const { return &_world; }
    void selectEntity(int index);
    void deselectEntity();
    bool selectedEntity(Entity_Handle *out) const;

public slots:
    void onTick(float deltaTime);
    void onRender(topo::IRender_Queue *rq);
    void setRunning(bool isRunning);
    void onMeshUpload(topo::IInstance *rq);
    void setGizmoMode(Session_Gizmo_Mode mode);

signals:
    void plantCreationAvailabilityChanged(bool isPlantCreationAvailable);
	void meshgenAvailabilityChanged(bool isMeshgenAvailableForSelectedEntity);

private:
    std::string _name;
    QWorld _world;
    bool _isRunning = false;
    std::optional<Entity_Handle> _selectedEntity;

    std::vector<Entity_Handle> _pendingColliderMeshUploads;

    glm::mat4 _matView;
    glm::mat4 _matProj;
    Session_Gizmo_Mode _gizmoMode = Session_Gizmo_Mode::Translation;

    std::optional<trigen::Session> _session;
};