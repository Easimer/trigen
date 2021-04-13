// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: entry point
//

#include "stdafx.h"
#include "wnd_main.h"
#include "renderer.h"
#include <QApplication>

int main(int argc, char **argv) {
	QApplication app(argc, argv);

	auto vm = std::make_unique<VM_Main>();
	Window_Main wnd(std::move(vm));
	wnd.show();

	return QApplication::exec();
}