// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <memory>
#include <QObject>

#include "session.h"
#include "renderer.h"
#include <softbody.h>

class VM_Main : public QObject {
	Q_OBJECT;
public:
	Session *session();
	Session *createNewSession(char const *name);
	void closeSession(Session *session);
	void switchToSession(Session *session);

	void addSoftbodySimulation(sb::Config const &cfg);
	void setRenderer(Renderer *renderer) {
		_renderer = renderer;
		_factory = Filament_Factory(renderer->engine());
	}

public slots:
	void onWindowResize(int w, int h);
	void onMouseDown(int x, int y);
	void onMouseUp(int x, int y);
	void onMouseWheel(int y);
	void onMouseMove(int x, int y);

signals:
	void cameraUpdated(filament::math::float3 const &eye, filament::math::float3 const &center);
	void currentSessionChanged(Session *session);

private:
	Renderer *_renderer = nullptr;
	std::list<std::unique_ptr<Session>> _sessions;
	Session *_currentSession = nullptr;
	Filament_Factory _factory;
};