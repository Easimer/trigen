// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: dual contouring library API
//

#pragma once

#include <optional>
#include <functional>
#include <vector>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace dc {
    struct Volume_Function_Result {
        float value;
        glm::vec4 normal;
        glm::vec2 uv;
    };

    using Volume_Function = std::function<std::optional<Volume_Function_Result>(glm::vec3 const& p)>;

    struct Parameters {
    };

    struct Mesh {
    };

    Mesh dual_contour(Parameters const& params, Volume_Function const& function);
}
