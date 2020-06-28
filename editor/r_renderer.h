// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#pragma once

#include <cstdint>
#include <vector>
#include <SDL_events.h>

namespace gfx {
    class IRenderer {
    public:
        virtual void new_frame() = 0;
        virtual double present() = 0;

        virtual void set_camera(Mat4 const& view_matrix) = 0;

        virtual void draw_points(Vec3 const* pPoints, size_t nCount, Vec3 const& vWorldPosition) = 0;

        virtual void draw_lines(
            Vec3 const* pEndpoints,
            size_t nLineCount,
            Vec3 const& vWorldPosition,
            Vec3 const& vStartColor,
            Vec3 const& vEndColor
        ) = 0;

        virtual bool pump_event_queue(SDL_Event& ev) = 0;
    };

    IRenderer* make_renderer();
    void destroy_renderer(IRenderer*);
}
