// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <memory>
#include <QObject>

#include "session.h"
#include <trigen.hpp>
#include <r_renderer.h>
#include "entity_list.h"

class VM_Main : public QObject {
	Q_OBJECT;
public:
	VM_Main(Entity_List_Model *entityListModel);
	Session *session();
	Session *createNewSession(char const *name);
	void closeSession(Session *session);
	void switchToSession(Session *session);

	void addSoftbodySimulation(Trigen_Parameters const &cfg);
	void addColliderFromPath(char const *path);
	void onTick(float deltaTime);
	void setGizmoMode(Session_Gizmo_Mode mode);
    void createMeshgenDialog(QWidget *parent);

public slots:
	void onRender(gfx::Render_Queue *rq);
	void setRunning(bool isRunning);
	void entitySelectionChanged(QModelIndex const &idx);

signals:
	void cameraUpdated();
	void currentSessionChanged(Session *session);
	void rendering(gfx::Render_Queue *rq);
    void meshgenAvailabilityChanged(bool isMeshgenAvailableForSelectedEntity);

private:
	std::list<std::unique_ptr<Session>> _sessions;
	Session *_currentSession = nullptr;
	Entity_List_Model *_entityListModel;

	gfx::Framebuffer_ID _framebuffer = nullptr;
};
