// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"
#include <cassert>
#include "vm_main.h"
#include "dlg_meshgen.h"

VM_Main::VM_Main(Entity_List_Model *entityListModel)
: _entityListModel(entityListModel) {
}

Session *VM_Main::session() {
	return _currentSession;
}

Session *VM_Main::createNewSession(char const *name) {
	auto session = std::make_unique<Session>(name);
	auto sessionPtr = session.get();

	switchToSession(sessionPtr);
	_sessions.emplace_back(std::move(session));

	return sessionPtr;
}

void VM_Main::closeSession(Session *session) {
}

void VM_Main::switchToSession(Session *session) {
	_currentSession = session;

	if (_currentSession != nullptr) {
		_entityListModel->setCurrentWorld(_currentSession->world());

        connect(_currentSession, &Session::meshgenAvailabilityChanged, this, &VM_Main::meshgenAvailabilityChanged);
	}

	emit currentSessionChanged(session);
}

void VM_Main::addSoftbodySimulation(Trigen_Parameters const &cfg) {
	if (_currentSession != nullptr) {
		_currentSession->createPlant(cfg);
	}
}

void VM_Main::addColliderFromPath(char const *path) {
	if (_currentSession != nullptr) {
		_currentSession->addColliderFromPath(path);
	}
}

void VM_Main::onTick(float deltaTime) {
	if (_currentSession != nullptr) {
		_currentSession->onTick(deltaTime);
	}
}

void VM_Main::setGizmoMode(Session_Gizmo_Mode mode) {
	if (_currentSession != nullptr) {
		_currentSession->setGizmoMode(mode);
	}
}

void VM_Main::createMeshgenDialog(QWidget *parent) {
	if (_currentSession != nullptr) {
        Entity_Handle selectedEntity;
		if (_currentSession->selectedEntity(&selectedEntity)) {
            auto dlg = make_meshgen_dialog(_currentSession->world(), selectedEntity, parent);
            dlg->show();
            connect(this, &VM_Main::rendering, dlg, &Base_Dialog_Meshgen::onRender);
		}
	}
}

void VM_Main::onRender(gfx::Render_Queue *rq) {
	if (_currentSession != nullptr) {
		_currentSession->onMeshUpload(rq);
		_currentSession->onRender(rq);
	}

    emit rendering(rq);
}

void VM_Main::setRunning(bool isRunning) {
	if (_currentSession != nullptr) {
		_currentSession->setRunning(isRunning);
	}
}

void VM_Main::entitySelectionChanged(QModelIndex const &idx) {
	if (_currentSession != nullptr) {
		if (idx.isValid()) {
			_currentSession->selectEntity(idx.row());
		} else {
			_currentSession->deselectEntity();
		}
	}
}
