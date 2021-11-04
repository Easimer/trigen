// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: entry point
//

#include "common.h"
#include "wnd_main.h"
#include <QApplication>
#include <imgui.h>

int main(int argc, char** argv) {
    QSurfaceFormat fmt;
    fmt.setVersion(4, 4);
    fmt.setProfile(QSurfaceFormat::CoreProfile);
    QSurfaceFormat::setDefaultFormat(fmt);

    auto *ctx = ImGui::CreateContext();
    ImGui::SetCurrentContext(ctx);

    QApplication app(argc, argv);

    Window_Main wnd;
    wnd.setAnimated(true);
    wnd.show();
    return app.exec();
}