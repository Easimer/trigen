// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#pragma once

#include <cstdint>
#include <vector>
#include <SDL_events.h>
#include <optional>

namespace gfx {
    struct Renderer_Config {
        unsigned width, height;
    };

    struct Render_Context_Supplement {
        // Position of the sun
        std::optional<Vec3> sun;
    };

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

        virtual void draw_ellipsoids(
            Render_Context_Supplement const& ctx,
            size_t count,
            Vec3 const* centers,
            Vec3 const* sizes,
            Quat const* rotations,
            Vec3 const& color = Vec3(0.6, 0.6, 0.6)
        ) = 0;

        virtual bool pump_event_queue(SDL_Event& ev) = 0;

        virtual void change_resolution(unsigned* inout_width, unsigned* inout_height) = 0;
        virtual void get_resolution(unsigned* out_width, unsigned* out_height) = 0;
    };

    IRenderer* make_renderer(Renderer_Config const&);
    void destroy_renderer(IRenderer*);
}
