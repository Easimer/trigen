// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: OptiX kernel parameters
//

#ifndef PARAMS_H
#define PARAMS_H

#include <stdint.h>
#include <optix.h>

struct Params {
    float3 *ray_origins;
    float3 *ray_directions;

    float3 **normals;
    unsigned **normal_indices;

    OptixTraversableHandle handle;

    uint8_t *flags;
    unsigned *ray_index;
    float3 *xp;
    float3 *surf_normal;
    float *depth;
};

struct Sbt_Raygen {
};

struct Sbt_ClosestHit {
};

struct Sbt_Miss {
};

#endif /* PARAMS_H */
