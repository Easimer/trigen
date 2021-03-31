// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"
#include "session.h"
#include <glm/gtc/type_ptr.hpp>

Session::Session() : _camera(create_arcball_camera()) {
}

std::unique_ptr<Session> Session::create() {
	struct U : Session {};
	return std::make_unique<U>();
}

void Session::onMouseDown(int x, int y) {
	_camera->mouse_down(x, y);
	
}

void Session::onMouseUp(int x, int y) {
	_camera->mouse_up(x, y);
}

void Session::onMouseWheel(int y) {
	_camera->mouse_wheel(y);
}

void Session::onWindowResize(int w, int h) {
	_camera->set_screen_size(w, h);
}

void Session::emitViewMatrixUpdated() {
	auto mat = _camera->get_view_matrix();
	// Both filament and GLM use column-major matrices
	auto &matPtr = *(filament::math::mat4f *)glm::value_ptr(mat);
	
	emit viewMatrixUpdated(matPtr);
}