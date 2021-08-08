// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <memory>
#include <QObject>

#include "session.h"
#include <trigen.hpp>
#include <topo.h>
#include "entity_list.h"

class VM_Main : public QObject {
    Q_OBJECT;

public:
    VM_Main(Entity_List_Model *entityListModel);
    void
    init(topo::IInstance *renderer);
    Session *
    session();
    Session *
    createNewSession(char const *name);
    void
    closeSession(Session *session);
    void
    switchToSession(Session *session);

    void
    addSoftbodySimulation(Trigen_Parameters const &cfg);
    void
    addColliderFromPath(char const *path);
    void
    onTick(float deltaTime);
    void
    setGizmoMode(Session_Gizmo_Mode mode);
    void
    createMeshgenDialog(QWidget *parent);

public slots:
    void
    onRender(topo::IRender_Queue *rq);
    void
    setRunning(bool isRunning);
    void
    entitySelectionChanged(QModelIndex const &idx);

signals:
    void
    cameraUpdated();
    void
    currentSessionChanged(Session *session);
    /** Emitted when we're rendering opaque objects */
    void
    rendering(topo::IRender_Queue *rq);
    void
    meshgenAvailabilityChanged(bool isMeshgenAvailableForSelectedEntity);

private:
    topo::IInstance *_renderer;
    std::list<std::unique_ptr<Session>> _sessions;
    Session *_currentSession = nullptr;
    Entity_List_Model *_entityListModel;
};
