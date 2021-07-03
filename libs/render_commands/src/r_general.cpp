// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"
#include <glm/vec3.hpp>
#include <r_cmd/general.h>
#include <stb_image.h>

using Vec3 = glm::vec3;

void Render_Grid::execute(gfx::IRenderer *renderer) {
    glm::vec3 lines[] = {
        glm::vec3(0, 0, 0),
        glm::vec3(1, 0, 0),
        glm::vec3(0, 0, 0),
        glm::vec3(0, 1, 0),
        glm::vec3(0, 0, 0),
        glm::vec3(0, 0, 1),
    };

    renderer->draw_lines(lines + 0, 1, Vec3(0, 0, 0), Vec3(.35, 0, 0), Vec3(1, 0, 0));
    renderer->draw_lines(lines + 2, 1, Vec3(0, 0, 0), Vec3(0, .35, 0), Vec3(0, 1, 0));
    renderer->draw_lines(lines + 4, 1, Vec3(0, 0, 0), Vec3(0, 0, .35), Vec3(0, 0, 1));

    // render grid
    Vec3 grid[80];
    for (int i = 0; i < 20; i++) {
        auto base = 4 * i;
        grid[base + 0] = Vec3(i - 10, 0, -10);
        grid[base + 1] = Vec3(i - 10, 0, +10);
        grid[base + 2] = Vec3(-10, 0, i - 10);
        grid[base + 3] = Vec3(+10, 0, i - 10);
    }

    renderer->draw_lines(grid, 40, Vec3(0, 0, 0), Vec3(0.4, 0.4, 0.4), Vec3(0.4, 0.4, 0.4));
}

Render_Model::Render_Model(gfx::Model_ID model, gfx::Texture_ID diffuse, gfx::Transform const &transform)
    : _model(model)
    , _diffuse(diffuse)
    , _normal(nullptr)
    , _transform(transform) {
}

Render_Model::Render_Model(gfx::Model_ID model, gfx::Texture_ID diffuse, gfx::Texture_ID normal, gfx::Transform const &transform)
    : _model(model)
    , _diffuse(diffuse)
    , _normal(normal)
    , _transform(transform) {
}

void Render_Model::execute(gfx::IRenderer *renderer) {
    if (_normal != nullptr) {
        gfx::Material_Lit material;
        material.diffuse = _diffuse;
        material.normal = _normal;
        renderer->draw_textured_triangle_elements(_model, material, _transform);
    } else {
        gfx::Material_Unlit material;
        material.diffuse = _diffuse;
        renderer->draw_textured_triangle_elements(_model, material, _transform);
    }
}

Render_Transparent_Model::Render_Transparent_Model(
    gfx::Model_ID model,
    gfx::Texture_ID diffuse,
    gfx::Transform const &transform)
    : _model(model)
    , _diffuse(diffuse)
    , _transform(transform) { }

void
Render_Transparent_Model::execute(gfx::IRenderer *renderer) {
    renderer->draw_transparent_model(_model, _diffuse, _transform);
}

Render_Untextured_Model::Render_Untextured_Model(
    gfx::Model_ID model,
    gfx::Transform const &transform)
    : _model(model)
    , _transform(transform)
    , _tintColor({ 1, 1, 1, 1 })
    , _wireframeOnTop(false) { }

void
Render_Untextured_Model::execute(gfx::IRenderer *renderer) {
    gfx::Render_Parameters params;
    params.tint_color = _tintColor;
    params.wireframe_on_top = _wireframeOnTop;

    renderer->draw_triangle_elements(params, _model, _transform);
}

Load_Texture_Command::Load_Texture_Command(std::optional<gfx::Texture_ID> *handle, void const *image, size_t image_len)
: _handle(handle), _image(image), _image_len(image_len) {
}

void Load_Texture_Command::execute(gfx::IRenderer *renderer) {
    if (!_handle->has_value()) {
        gfx::Texture_ID id;
        int width, height;
        int channels;
        auto data = stbi_load_from_memory((stbi_uc *)_image, _image_len, &width, &height, &channels, 3);
        if (renderer->upload_texture(&id, width, height, gfx::Texture_Format::RGB888, data)) {
            _handle->emplace(id);
        }
        stbi_image_free(data);
    }
}

Destroy_Model_Command::Destroy_Model_Command(gfx::Model_ID id)
: _id(id) {
}

void Destroy_Model_Command::execute(gfx::IRenderer *renderer) {
    renderer->destroy_model(_id);
}

Fetch_Camera_Matrices::Fetch_Camera_Matrices(glm::mat4 *out_viewMatrix, glm::mat4 *out_projectionMatrix)
: _viewMatrix(out_viewMatrix), _projMatrix(out_projectionMatrix) {
}

void Fetch_Camera_Matrices::execute(gfx::IRenderer *renderer) {
    renderer->get_camera(*_viewMatrix, *_projMatrix);
}

Upload_Texture_Command::Upload_Texture_Command(gfx::Texture_ID *outHandle, void const *buffer, int width, int height, gfx::Texture_Format format)
    : _outHandle(outHandle)
    , _buffer(buffer)
    , _width(width)
    , _height(height)
    , _format(format) {
    assert(_outHandle != nullptr);
    assert(_buffer != nullptr);
    assert(_width > 0);
    assert(_height > 0);
}

void Upload_Texture_Command::execute(gfx::IRenderer *renderer) {
    assert(*_outHandle == nullptr);
    if (_outHandle == nullptr || *_outHandle != nullptr || _buffer == nullptr || _width == 0 || _height == 0) {
        return;
    }

    renderer->upload_texture(_outHandle, _width, _height, _format, _buffer);
}

Create_Framebuffer_Command::Create_Framebuffer_Command(
    gfx::Framebuffer_ID *outHandle,
    float resolution_scale)
    : _outHandle(outHandle)
    , _resolution_scale(resolution_scale) { }

void
Create_Framebuffer_Command::execute(gfx::IRenderer *renderer) {
    renderer->create_framebuffer(_outHandle, _resolution_scale);
}

Destroy_Framebuffer_Command::Destroy_Framebuffer_Command(
    gfx::Framebuffer_ID handle)
    : _handle(handle) { }

void
Destroy_Framebuffer_Command::execute(gfx::IRenderer *renderer) {
    renderer->destroy_framebuffer(_handle);
}

Activate_Framebuffer_Command::Activate_Framebuffer_Command(
    gfx::Framebuffer_ID handle) : _handle(handle) { }

void
Activate_Framebuffer_Command::execute(gfx::IRenderer *renderer) {
    renderer->activate_framebuffer(_handle);
}

Draw_Framebuffer_Command::Draw_Framebuffer_Command(gfx::Framebuffer_ID handle)
    : _handle(handle) { }

void
Draw_Framebuffer_Command::execute(gfx::IRenderer *renderer) {
    renderer->draw_framebuffer(_handle);
}

Clear_Command::Clear_Command(glm::vec4 color)
    : _color(color) { }

void
Clear_Command::execute(gfx::IRenderer *renderer) {
    renderer->clear(_color);
}
