// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: OptiX kernels
//

#include <stdio.h>
#include <math.h>
#include "params.h"
#include <optix.h>
#include <optix_device.h>
#include <cuda_runtime.h>

extern "C" {
    __constant__ struct Params params;
}

enum Trace_Flags {
    ETRACE_NONE = 0,
    // The ray has hit something
    ETRACE_HIT = 1 << 0,
};

inline __device__ float3 operator+(float3 lhs, float3 rhs) {
    return { lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z };
}

inline __device__ float3 operator-(float3 lhs, float3 rhs) {
    return { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z };
}

inline __device__ float3 operator/(float3 lhs, float rhs) {
    return { lhs.x / rhs, lhs.y / rhs, lhs.z / rhs };
}

inline __device__ float3 operator*(float lhs, float3 rhs) {
    return { lhs * rhs.x, lhs * rhs.y, lhs * rhs.z };
}

inline __device__ float length(float3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline __device__ float3 normalized(float3 v) {
    float len = length(v);
    return { v.x / len, v.y / len, v.z / len };
}

#define MAKE_FLOAT3_PAYLOAD(x, y, z) make_float3(int_as_float(x), int_as_float(y), int_as_float(z))

extern "C"
__global__ void __raygen__intersect() {
    uint3 const idx = optixGetLaunchIndex();
    uint3 const dim = optixGetLaunchDimensions();

    params.ray_index[idx.x] = idx.x;

    // RAY PAYLOAD:
    // Flags (Trace_Flags)
    unsigned int p_flags;
    // Intersection point
    unsigned int p_xp0, p_xp1, p_xp2;
    // Surface normal
    unsigned int p_n0, p_n1, p_n2;
    // Penetration depth
    unsigned int p_depth;

    float3 ray_origin = params.ray_origins[idx.x];
    float3 ray_direction = params.ray_directions[idx.x];

    optixTrace(
        params.handle,
        ray_origin,
        ray_direction,
        0.0f,
        1.0f,
        0.0f,
        OptixVisibilityMask(0xFF),
        OPTIX_RAY_FLAG_NONE,
        0,
        0,
        0,
        p_flags,
        p_xp0, p_xp1, p_xp2,
        p_n0, p_n1, p_n2,
        p_depth
    );

    params.flags[idx.x] = p_flags;
    params.depth[idx.x] = int_as_float(p_depth);
    params.xp[idx.x] = MAKE_FLOAT3_PAYLOAD(p_xp0, p_xp1, p_xp2);
    params.surf_normal[idx.x] = MAKE_FLOAT3_PAYLOAD(p_n0, p_n1, p_n2);
}

extern "C"
__global__ void __miss__intersect() {
    // Set flags register to zero
    optixSetPayload_0(ETRACE_NONE);
    optixSetPayload_1(optixUndefinedValue());
    optixSetPayload_2(optixUndefinedValue());
    optixSetPayload_3(optixUndefinedValue());
    optixSetPayload_4(optixUndefinedValue());
    optixSetPayload_5(optixUndefinedValue());
    optixSetPayload_6(optixUndefinedValue());
    optixSetPayload_7(optixUndefinedValue());
}

extern "C"
__global__ void __closesthit__intersect() {
    float t = optixGetRayTmax();
    unsigned const idx = optixGetLaunchIndex().x;

    // NOTE: not needed since we set Tmax to 1 in the call to optixTrace
    // if (t < 0 || 1 < t) {
    //     return;
    // }

    // Set hit flag in flags register
    optixSetPayload_0(ETRACE_HIT);

    float3 x0 = optixGetWorldRayOrigin();
    float3 d = optixGetWorldRayDirection();
    float3 x1 = x0 + d;

    // Calculate intersection point and depth
    float3 xp = x0 + t * d;

    // Calculate penetration depth
    float depth = length(x1 - xp);
    optixSetPayload_7(float_as_int(depth));

    optixSetPayload_1(float_as_int(xp.x));
    optixSetPayload_2(float_as_int(xp.y));
    optixSetPayload_3(float_as_int(xp.z));

    // TODO: get instance ID, use that to index into the normals array
    // OptixTraversableHandle gas = optixGetGASTraversableHandle();
    unsigned instance_id = optixGetInstanceId();
    unsigned primitive_idx = optixGetPrimitiveIndex();
    // unsigned sbt_idx = optixGetSbtGASIndex();

    float3 *normals = params.normals[instance_id];
    unsigned *normal_indices = params.normal_indices[instance_id];

    float3 n0, n1, n2;
    unsigned ni0, ni1, ni2;
    ni0 = normal_indices[primitive_idx * 3 + 0];
    ni1 = normal_indices[primitive_idx * 3 + 1];
    ni2 = normal_indices[primitive_idx * 3 + 2];
    n0 = normals[ni0];
    n1 = normals[ni1];
    n2 = normals[ni2];

    float3 normal = (n0 + n1 + n2) / 3.0f;
    normal = optixTransformNormalFromObjectToWorldSpace(normal);
    optixSetPayload_4(float_as_int(normal.x));
    optixSetPayload_5(float_as_int(normal.y));
    optixSetPayload_6(float_as_int(normal.z));
}

// vim: et:ts=4:sw=4
