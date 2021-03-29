// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: entry point
//

#include "stdafx.h"
#include "wnd_main.h"
#include "filament_viewport.h"
#include <QApplication>

int main(int argc, char **argv) {
	QApplication app(argc, argv);

	Window_Main wnd;
	Filament_Viewport viewport(&wnd, filament::Engine::Backend::VULKAN);

	viewport.postInit();
	wnd.setViewport(&viewport);
	wnd.show();

	return app.exec();
}