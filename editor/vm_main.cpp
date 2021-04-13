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

void VM_Main::onRender(gfx::Render_Queue *rq) {
	if (_currentSession != nullptr) {
		_currentSession->onRender(rq);
	}
}