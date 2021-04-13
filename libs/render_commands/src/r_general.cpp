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

Render_Model::Render_Model(gfx::Model_ID model, gfx::Texture_ID diffuse, gfx::Transform const &transform) :
    _model(model), _diffuse(diffuse), _transform(transform) {
}

void Render_Model::execute(gfx::IRenderer *renderer) {
    gfx::Material_Unlit material{};
    material.diffuse = _diffuse;

    renderer->draw_textured_triangle_elements(_model, material, _transform);
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
