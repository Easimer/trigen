// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#pragma once

#include <cstdint>
#include <vector>
#include <optional>
#include <memory>

#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/quaternion.hpp>

namespace gfx {
    struct Render_Context_Supplement {
        // Position of the sun
        std::optional<glm::vec3> sun;
    };

    class IRenderer {
    public:
        virtual ~IRenderer() {}

        virtual void new_frame() = 0;
        virtual double present() = 0;

        virtual void set_camera(glm::mat4 const& view_matrix) = 0;

        virtual void draw_points(size_t nCount, glm::vec3 const* pPoints, glm::vec3 const& vWorldPosition) = 0;

        virtual void draw_lines(
            glm::vec3 const* pEndpoints,
            size_t nLineCount,
            glm::vec3 const& vWorldPosition,
            glm::vec3 const& vStartColor,
            glm::vec3 const& vEndColor
        ) = 0;

        virtual void draw_ellipsoids(
            Render_Context_Supplement const& ctx,
            size_t count,
            glm::vec3 const* centers,
            glm::vec3 const* sizes,
            glm::quat const* rotations,
            glm::vec3 const& color = glm::vec3(0.6, 0.6, 0.6)
        ) = 0;

        virtual void change_resolution(unsigned* inout_width, unsigned* inout_height) = 0;
        virtual void get_resolution(unsigned* out_width, unsigned* out_height) = 0;
    };

    enum class Renderer_Backend {
        OpenGL,
    };

    std::unique_ptr<IRenderer> make_opengl_renderer(void* glctx, void* (*getProcAddress)(char const*));
}