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

	auto vm = std::make_unique<VM_Main>();
	// Retain a pointer to the VM so we can call setRenderer on it below
	auto vm_ptr = vm.get();
	Window_Main wnd(std::move(vm));
	Filament_Viewport viewport(&wnd);

	auto nativeHandle = viewport.winId();
	Renderer renderer(filament::Engine::Backend::VULKAN, (void *)nativeHandle);
	viewport.setRenderer(&renderer);
	vm_ptr->setRenderer(&renderer);

	wnd.setViewport(&viewport);
	wnd.show();

	return QApplication::exec();
}