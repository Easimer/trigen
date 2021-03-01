// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#include "uv_unwrap.h"

int PSP::paint(/* out */ Material &material, /* inout */ Mesh &mesh) {
    // Assign UV coordinates
    auto charts = chart(mesh);
    project_charts(mesh, charts);
    sort_charts_by_area_descending(mesh, charts);
    divide_quad_among_charts(mesh, charts);
    // Setup particle system
    // Simulate particles:
    //  - calculate next position based on velocity
    //  - check collisions; for all triangle: collision(triangle, ray(next, current))
    //    - on collision, apply brush to the material
    return 0;
}