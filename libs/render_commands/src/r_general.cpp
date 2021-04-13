// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"
#include <glm/vec3.hpp>
#include <r_cmd/general.h>

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
