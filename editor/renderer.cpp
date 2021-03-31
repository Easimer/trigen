// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: renderer
//

#include "stdafx.h"
#include "renderer.h"

#include <filament/Camera.h>
#include <filament/Engine.h>
#include <filament/IndexBuffer.h>
#include <filament/Material.h>
#include <filament/MaterialInstance.h>
#include <filament/RenderableManager.h>
#include <filament/Scene.h>
#include <filament/Skybox.h>
#include <filament/TransformManager.h>
#include <filament/VertexBuffer.h>
#include <filament/View.h>
#include <filament/SwapChain.h>
#include <filament/Fence.h>
#include <filament/Renderer.h>
#include <filament/Viewport.h>

Renderer::Renderer(filament::Engine::Backend backend, void *nativeHandle) {
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

void Renderer::onClose() {
	filament::Fence::waitAndDestroy(_engine->createFence());
}

void Renderer::draw() {
	if (_renderer->beginFrame(_swapChain)) {
		_renderer->render(_view);
		_renderer->endFrame();
	}
}

void Renderer::updateCameraProjection(uint32_t w, uint32_t h) {
	_view->setViewport({ 0, 0, w, h });
}