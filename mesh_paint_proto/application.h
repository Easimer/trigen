// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <r_sdl.h>
#include <memory>

class IApplication {
public:
    virtual ~IApplication() = default;

    virtual int run() = 0;
};

std::unique_ptr<IApplication> make_application(
    std::unique_ptr<gfx::ISDL_Window> &&window
);
