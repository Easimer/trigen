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
    struct Render_Parameters {
        // Position of the sun
        std::optional<glm::vec3> sun;

        // Mesh surface tint color
        std::optional<glm::vec4> tint_color;
    };

    using Model_ID = void*;
    using Texture_ID = void*;

    struct Model_Descriptor {
        size_t vertex_count = 0;
        std::array<float, 3> const *vertices = nullptr;
        std::array<float, 3> const *normals = nullptr;
        std::array<float, 2> const *uv = nullptr;
        size_t element_count = 0;
        unsigned const *elements = nullptr;
    };

    enum class Texture_Format {
        RGB888,
    };

    struct Material_Unlit {
        Texture_ID diffuse;
    };

    struct Material_Lit {
        Texture_ID diffuse;
        Texture_ID normal;
    };

    struct Transform {
        glm::vec3 position;
        glm::quat rotation;
        glm::vec3 scale;
    };

    class IRenderer {
    public:
        virtual ~IRenderer() {}

        virtual void new_frame() = 0;
        virtual double present() = 0;

        virtual void set_camera(glm::mat4 const& view_matrix) = 0;
        virtual void get_camera(glm::mat4 &view_matrix, glm::mat4 &projection_matrix) = 0;

        virtual void set_sun_position(glm::vec3 const &position) = 0;

        virtual void draw_points(size_t nCount, glm::vec3 const* pPoints, glm::vec3 const& vWorldPosition) = 0;

        virtual void draw_lines(
            glm::vec3 const* pEndpoints,
            size_t nLineCount,
            glm::vec3 const& vWorldPosition,
            glm::vec3 const& vStartColor,
            glm::vec3 const& vEndColor
        ) = 0;

        virtual void draw_triangle_elements(
            size_t vertex_count,
            std::array<float, 3> const* vertices,
            size_t element_count,
            unsigned const* elements,
            glm::vec3 const& vWorldPosition
        ) = 0;

        virtual void draw_triangle_elements_with_vertex_color(
            size_t vertex_count,
            std::array<float, 3> const* vertices,
            glm::u8vec3 const* vertex_colors,
            size_t element_count,
            unsigned const* elements,
            glm::vec3 const& vWorldPosition
        ) = 0;

        virtual void change_resolution(unsigned* inout_width, unsigned* inout_height) = 0;
        virtual void get_resolution(unsigned* out_width, unsigned* out_height) = 0;

        virtual bool upload_texture(Texture_ID *out_id, unsigned width, unsigned height, Texture_Format format, void const *image) = 0;
        virtual void destroy_texture(Texture_ID id) = 0;

        virtual bool create_model(Model_ID *out_id, Model_Descriptor const *model) = 0;
        virtual void destroy_model(Model_ID id) = 0;

        virtual void draw_textured_triangle_elements(
            Model_ID model,
            Material_Unlit const &material,
            Transform const &transform
        ) = 0;

        virtual void draw_textured_triangle_elements(
            Model_ID model,
            Material_Lit const &material,
            Transform const &transform
        ) = 0;

        virtual void draw_triangle_elements(
            gfx::Model_ID model_handle,
            gfx::Transform const &transform
        ) = 0;

        virtual void draw_triangle_elements(
            Render_Parameters const &params,
            gfx::Model_ID model_handle,
            gfx::Transform const &transform
        ) = 0;
    };

    enum class Renderer_Backend {
        OpenGL,
    };

    std::unique_ptr<IRenderer> make_opengl_renderer(void* glctx, void* (*getProcAddress)(char const*));
}
