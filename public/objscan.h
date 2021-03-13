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
    float subdivisions;              // how many subdivisions to do
    
    // Debug information (filled in by objscan)
    objscan_position bb_min, bb_max; // bounding box
    unsigned threads_used;           // how many threads were used
    float step_x, step_y, step_z;    // subdivision step sizes
};

struct objscan_result {
    // Extra, optional parameters and diagnostic informations; can be NULL
    objscan_extra* extra = nullptr;
    
    // Number of elements in `positions`
    unsigned long long particle_count = 0;
    // Particle positions
    objscan_position* positions = nullptr;
    
    // Number of elements in `connections`
    unsigned long long connection_count = 0;
    // Particle connections
    // Note that this list may contain duplicate pairs.
    objscan_connection* connections = nullptr;
};

// Loads a Wavefront OBJ model and turns it into a particle system
//
// @param res  Results will be placed here, can't be NULL.
// @param path Path to the OBJ file on disk, can't be NULL.
// @return A value indicating whether the conversion succeeded or not.
bool objscan_from_obj_file(objscan_result* res, char const* path);

// Frees the resources associated with this result structure.
// NOTE: This does not free the objscan_result struct itself.
//
// @param res Pointer to the results struct, can't be NULL.
void objscan_free_result(objscan_result* res);