// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#include "uv_unwrap.h"

int PSP::unwrap(/* inout */ Mesh &mesh) {
    // Assign UV coordinates
    auto charts = chart(mesh);
    split_vertices(mesh, charts);
    project_charts(mesh, charts);
    sort_charts_by_area_descending(mesh, charts);
    divide_quad_among_charts(mesh, charts);
    return 0;
}