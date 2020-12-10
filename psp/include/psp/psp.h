// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

namespace PSP {
    struct Mesh {
        // Filled in by you
        std::vector<glm::vec3> position;
        std::vector<glm::vec3> normal;
        std::vector<size_t> elements;

        // Filled in by us and must be empty
        std::vector<glm::vec2> uv;
        std::vector<glm::u8vec3> chart_debug_color;
    };

    struct Texture {
        int width, height;
        std::unique_ptr<glm::vec<3, float>[]> data;
    };

    struct Material {
        Texture albedo;
        Texture normal;
    };

    int paint(/* out */ Material &material, /* inout */ Mesh &mesh);
}
