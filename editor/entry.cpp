// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: entry point
//

#include "stdafx.h"
#include "wnd_main.h"
#include <QApplication>
#include <imgui.h>
#include "entity_list.h"

#include "trigen_worker.h"

static int runApplication(int argc, char **argv) {
	QApplication app(argc, argv);
	qRegisterMetaType<Stage_Tag>();
	qRegisterMetaType<Trigen_Session>();
	qRegisterMetaType<Trigen_Status>();
	qRegisterMetaType<std::function<Trigen_Status(Trigen_Session)>>();

	auto entityListModel = std::make_unique<Entity_List_Model>();
	auto vm = std::make_unique<VM_Main>(entityListModel.get());
	Window_Main wnd(std::move(vm), std::move(entityListModel));
	wnd.show();

	return QApplication::exec();
}

int main(int argc, char **argv) {
	int rc;

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	rc = runApplication(argc, argv);

	ImGui::DestroyContext();

	return rc;
}