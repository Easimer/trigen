// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: renderer declaration
//

#pragma once

#include <filament/Engine.h>
#include <math/mat4.h>
#include <utils/Entity.h>
#include "filament_factory.h"

class Renderer {
public:
	Renderer(filament::Engine::Backend backend, void *nativeHandle);

	void onClose();
	void draw();
	void updateCameraProjection(uint32_t w, uint32_t h);
	void updateCamera(filament::math::float3 const &eye, filament::math::float3 const &center);
	filament::Engine *engine() { return _engine; }
	void setScene(filament::Scene *scene);

private:
	void *_surfaceNativeHandle;
	Filament_Factory _factory;
	filament::Engine *_engine = nullptr;
	filament::SwapChain *_swapChain = nullptr;
	filament::View *_view = nullptr;
	filament::Renderer *_renderer = nullptr;
	filament::Camera *_camera = nullptr;
	filament::Skybox *_skybox = nullptr;
	filament::Scene *_scene = nullptr;

	// TEMP:
	filament::VertexBuffer *_vb;
	filament::IndexBuffer *_ib;
	filament::Material *_mat;
	utils::Entity _renderable;
};