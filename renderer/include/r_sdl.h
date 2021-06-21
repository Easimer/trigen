// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#pragma once

#include "r_renderer.h"
#include <SDL_events.h>

namespace gfx {
    struct Surface_Config {
        unsigned width, height;
        char const* title;
    };

    class RENDERER_EXPORT ISDL_Window : public IRenderer {
    public:
        virtual bool poll_event(SDL_Event* ev) = 0;
    };

    RENDERER_EXPORT std::unique_ptr<ISDL_Window> make_window(Surface_Config const& cfg, Renderer_Backend backend);
}
