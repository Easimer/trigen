// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: renderer declaration
//

#pragma once

#include <filament/Engine.h>
#include <filament/SwapChain.h>
#include <filament/Fence.h>
#include <filament/View.h>
#include <filament/Renderer.h>
#include <filament/Viewport.h>
#include <filament/Camera.h>
#include <filament/Skybox.h>
#include <filament/Scene.h>

class Renderer {
public:
	Renderer(filament::Engine::Backend backend, void *nativeHandle) {
		_engine = filament::Engine::create(backend);

		_swapChain = _engine->createSwapChain(nativeHandle);
		_view = _engine->createView();
		_renderer = _engine->createRenderer();
		_camera = _engine->createCamera();

		_view->setCamera(_camera);

		_skybox = filament::Skybox::Builder().color({ 0.1, 0.125, 0.25, 1.0 }).build(*_engine);
		_scene = _engine->createScene();
		_scene->setSkybox(_skybox);
		_view->setScene(_scene);
	}

	void onClose() {
		filament::Fence::waitAndDestroy(_engine->createFence());
	}

	void draw() {
		if (_renderer->beginFrame(_swapChain)) {
			_renderer->render(_view);
			_renderer->endFrame();
		}
	}

	void updateCameraProjection(uint32_t w, uint32_t h) {
		_view->setViewport({ 0, 0, w, h });
	}

private:
	filament::Engine *_engine = nullptr;
	filament::SwapChain *_swapChain = nullptr;
	filament::View *_view = nullptr;
	filament::Renderer *_renderer = nullptr;
	filament::Camera *_camera = nullptr;
	filament::Skybox *_skybox = nullptr;
	filament::Scene *_scene = nullptr;
};