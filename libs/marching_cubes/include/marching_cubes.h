// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#pragma once

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <vector>

namespace marching_cubes {
    struct metaball {
        glm::vec3 position;
        float radius;
        float scale;
    };

    struct mesh {
        std::vector<glm::vec3> positions;
        std::vector<glm::vec3> normal;
        std::vector<unsigned> indices;
    };

    struct params {
        int subdivisions;
    };

    mesh generate(std::vector<metaball> const &metaballs, params const &params);
}
