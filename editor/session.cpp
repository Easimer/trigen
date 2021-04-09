// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"
#include "session.h"
#include "plant_component.h"
#include <glm/gtc/type_ptr.hpp>

Session::Session(char const *name) : _camera(create_arcball_camera()), _name(name), _world(_scene) {
}

void Session::createPlant(sb::Config const &cfg) {
	auto ent = _world.createEntity();
	_world.attachComponent<Plant_Component>(ent, cfg);
}

void Session::onMouseDown(int x, int y) {
	_camera->mouse_down(x, y);
}

void Session::onMouseUp(int x, int y) {
	_camera->mouse_up(x, y);
	emitCameraUpdated();
}

void Session::onMouseMove(int x, int y) {
	_camera->mouse_move(x, y);
	emitCameraUpdated();
}

void Session::onMouseWheel(int y) {
	_camera->mouse_wheel(y);
	emitCameraUpdated();
}

void Session::onWindowResize(int w, int h) {
	_camera->set_screen_size(w, h);
}

void Session::emitCameraUpdated() {
	glm::vec3 eye, center;
	filament::math::float3 feye, fcenter;

	_camera->get_look_at(eye, center);
	feye = { eye.x, eye.y, eye.z };
	fcenter = { center.x, center.y, center.z };

	emit cameraUpdated(feye, fcenter);
}
