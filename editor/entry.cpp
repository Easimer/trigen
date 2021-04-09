// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: entry point
//

#include "stdafx.h"
#include "wnd_main.h"
#include "filament_viewport.h"
#include "renderer.h"
#include <QApplication>

int main(int argc, char **argv) {
	QApplication app(argc, argv);

	Window_Main wnd;
	Filament_Viewport viewport(&wnd);

	auto nativeHandle = viewport.winId();
	Renderer renderer(filament::Engine::Backend::VULKAN, (void *)nativeHandle);
	viewport.setRenderer(&renderer);


	wnd.setViewport(&viewport);
	wnd.show();

	return app.exec();
}