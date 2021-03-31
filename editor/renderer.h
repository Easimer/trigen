// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: renderer declaration
//

#pragma once

#include <filament/Engine.h>

class Renderer {
public:
	Renderer(filament::Engine::Backend backend, void *nativeHandle);

	void onClose();
	void draw();
	void updateCameraProjection(uint32_t w, uint32_t h);

private:
	filament::Engine *_engine = nullptr;
	filament::SwapChain *_swapChain = nullptr;
	filament::View *_view = nullptr;
	filament::Renderer *_renderer = nullptr;
	filament::Camera *_camera = nullptr;
	filament::Skybox *_skybox = nullptr;
	filament::Scene *_scene = nullptr;
};