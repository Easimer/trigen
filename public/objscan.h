// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: objscan API
//

#pragma once

struct objscan_connection {
    long long idx0, idx1;
};

struct objscan_position {
    float x, y, z, w;
};

struct objscan_extra {
    // Parameters (filled in by the caller)
    float subdivisions; // how many subdivisions to do
    
    // Debug information (filled in by objscan)
    objscan_position bb_min, bb_max; // bounding box
    unsigned threads_used;
    float step_x, step_y, step_z; // subdivision step sizes
};

struct objscan_result {
    objscan_extra* extra = nullptr;
    
    long long particle_count = 0;
    objscan_position* positions = nullptr;
    long long connection_count = 0;
    objscan_connection* connections = nullptr;
};

bool objscan_from_obj_file(objscan_result* res, char const* path);
void objscan_free_result(objscan_result* res);