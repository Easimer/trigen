// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"
#include "application.h"
#include <cstdio>

int main(int argc, char **argv) {
    gfx::Surface_Config surf = {};
    surf.title = "Mesh paint";
    surf.width = 1280;
    surf.height = 720;

    auto wnd = gfx::make_window(surf, gfx::Renderer_Backend::OpenGL);
    if (!wnd) {
        fprintf(stderr, "Failed to create window!\n");
        return 1;
    }

    auto app = make_application(std::move(wnd));

    return app->run();
}
