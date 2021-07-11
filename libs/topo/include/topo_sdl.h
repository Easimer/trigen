// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#pragma once

#include <topo.h>

#include <SDL_events.h>

namespace topo {
    struct Surface_Config {
        unsigned width, height;
        char const* title;
    };

    class TOPO_EXPORT ISDL_Window : public IInstance {
    public:
        ~ISDL_Window() override = default;

        virtual bool
        PollEvent(SDL_Event *ev)
            = 0;
    };

    TOPO_EXPORT UPtr<ISDL_Window> MakeWindow(Surface_Config const& cfg);
}
