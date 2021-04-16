// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"
#include <cassert>
#include "vm_main.h"

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

	emit currentSessionChanged(session);
}

void VM_Main::addSoftbodySimulation(sb::Config const &cfg) {
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

void VM_Main::onRender(gfx::Render_Queue *rq) {
	if (_currentSession != nullptr) {
		_currentSession->onMeshUpload(rq);
		_currentSession->onRender(rq);
	}
}

void VM_Main::setRunning(bool isRunning) {
	if (_currentSession != nullptr) {
		_currentSession->setRunning(isRunning);
	}
}