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
	assert(_renderer != nullptr);
	auto session = std::make_unique<Session>(name);
	auto sessionPtr = session.get();

	switchToSession(sessionPtr);
	_sessions.emplace_back(std::move(session));

	return sessionPtr;
}

void VM_Main::closeSession(Session *session) {
}

void VM_Main::switchToSession(Session *session) {
	if (_currentSession != nullptr) {
		disconnect(_currentSession, &Session::cameraUpdated, this, &VM_Main::cameraUpdated);
	}

	_currentSession = session;
	
	if (_currentSession != nullptr) {
		connect(_currentSession, &Session::cameraUpdated, this, &VM_Main::cameraUpdated);
	}

	emit currentSessionChanged(session);
}

void VM_Main::addSoftbodySimulation(sb::Config const &cfg) {
	if (_currentSession != nullptr) {
		_currentSession->createPlant(cfg);
	}
}

void VM_Main::onMouseDown(int x, int y) {
	if (_currentSession != nullptr) {
		_currentSession->onMouseDown(x, y);
	}
}

void VM_Main::onMouseUp(int x, int y) {
	if (_currentSession != nullptr) {
		_currentSession->onMouseUp(x, y);
	}
}

void VM_Main::onMouseWheel(int y) {
	if (_currentSession != nullptr) {
		_currentSession->onMouseWheel(y);
	}
}

void VM_Main::onMouseMove(int x, int y) {
	if (_currentSession != nullptr) {
		_currentSession->onMouseMove(x, y);
	}
}

void VM_Main::onWindowResize(int w, int h) {
	if (_currentSession != nullptr) {
		_currentSession->onWindowResize(w, h);
	}
}