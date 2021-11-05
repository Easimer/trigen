// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#pragma once

#include <psp/psp.h>

#include <vector>

using Triangle_ID = size_t;

struct Chart {
    int direction;
    bool is_degenerate;
    std::vector<Triangle_ID> triangles;
};

std::vector<Chart> chart(PSP::Mesh &mesh);
void project_charts(PSP::Mesh &mesh, std::vector<Chart> &charts);
void split_vertices(PSP::Mesh &mesh, std::vector<Chart> const &charts);
void sort_charts_by_area_descending(PSP::Mesh const &mesh, std::vector<Chart> &charts);
void divide_quad_among_charts(PSP::Mesh &mesh, std::vector<Chart> const &charts);
