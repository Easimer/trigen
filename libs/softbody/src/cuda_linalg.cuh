// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: CUDA linear algebra operations
//

#pragma once

#include "cuda_helper_math.h"

#define CUDA_LINALG_OP __inline__ __device__ __host__

CUDA_LINALG_OP void mat_add(float4* out, float4 const* lhs, float4 const* rhs) {
    out[0] = lhs[0] + rhs[0];
    out[1] = lhs[1] + rhs[1];
    out[2] = lhs[2] + rhs[2];
    out[3] = lhs[3] + rhs[3];
}

CUDA_LINALG_OP void mat_add_assign(float4* out, float4 const* other) {
    out[0] += other[0];
    out[1] += other[1];
    out[2] += other[2];
    out[3] += other[3];
}

CUDA_LINALG_OP void mat_sub_assign(float4* out, float4 const* other) {
    out[0] -= other[0];
    out[1] -= other[1];
    out[2] -= other[2];
    out[3] -= other[3];
}

CUDA_LINALG_OP void mat_scale(float s, float4* m) {
    m[0] = s * m[0];
    m[1] = s * m[1];
    m[2] = s * m[2];
    m[3] = s * m[3];
}

#define DEF_SWIZZLE_OP(name, c0, c1, c2, c3) \
CUDA_LINALG_OP float4 name(float4 v) { \
    return make_float4(v.c0, v.c1, v.c2, v.c3); \
}

#define DEF_SWIZZLE_OP_XYZ(name, c0, c1, c2) \
CUDA_LINALG_OP float3 name(float4 v) { \
    return make_float3(v.c0, v.c1, v.c2); \
}

#define DEF_SWIZZLE4_OP(name, c) DEF_SWIZZLE_OP(name, c, c, c, c)
DEF_SWIZZLE4_OP(xxxx, x)
DEF_SWIZZLE4_OP(yyyy, y)
DEF_SWIZZLE4_OP(zzzz, z)
DEF_SWIZZLE4_OP(wwww, w)
DEF_SWIZZLE_OP(xwww, x, w, w, w)
DEF_SWIZZLE_OP(wyww, w, y, w, w)
DEF_SWIZZLE_OP(wwzw, w, w, z, w)
DEF_SWIZZLE_OP_XYZ(xyz, x, y, z)

CUDA_LINALG_OP void mat_mul(float4* out, float4 const* lhs, float4 const* rhs) {
    for(int i = 0; i < 4; i++) {
        float4 sum = make_float4(0, 0, 0, 0);
        sum = xxxx(rhs[i]) * lhs[0] + sum;
        sum = yyyy(rhs[i]) * lhs[1] + sum;
        sum = zzzz(rhs[i]) * lhs[2] + sum;
        sum = wwww(rhs[i]) * lhs[3] + sum;
        out[i] = sum;
    }
}

CUDA_LINALG_OP void outer_product(float4* m, float4 p, float4 q) {
    m[0] = p * xxxx(q);
    m[1] = p * yyyy(q);
    m[2] = p * zzzz(q);
    m[3] = p * wwww(q);
}

CUDA_LINALG_OP void quat_to_mat(float4* m, float4 q) {
    float qxx = q.x * q.x;
    float qyy = q.y * q.y;
    float qzz = q.z * q.z;
    float qxz = q.x * q.z;
    float qxy = q.x * q.y;
    float qyz = q.y * q.z;
    float qwx = q.w * q.x;
    float qwy = q.w * q.y;
    float qwz = q.w * q.z;

    m[0].x = 1 - 2 * (qyy + qzz);
    m[0].y = 2 * (qxy + qwz);
    m[0].z = 2 * (qxz - qwy);
    m[0].w = 0;

    m[1].x = 2 * (qxy - qwz);
    m[1].y = 1 - 2 * (qxx + qzz);
    m[1].z = 2 * (qyz + qwx);
    m[1].w = 0;

    m[2].x = 2 * (qxz + qwy);
    m[2].y = 2 * (qyz - qwx);
    m[2].z = 1 - 2 * (qxx + qyy);
    m[2].w = 0;
    
    m[3] = make_float4(0, 0, 0, 0);
}

CUDA_LINALG_OP void diagonal3x3(float4* m, float4 diag) {
    diag.w = 0;
    m[0] = xwww(diag);
    m[1] = wyww(diag);
    m[2] = wwzw(diag);
    m[3] = make_float4(0, 0, 0, 0);
}

CUDA_LINALG_OP float4 hamilton_product(float4 q, float4 p) {
    auto w = q.w * p.w - q.x * p.x - q.y * p.y - q.z * p.z;
    auto i = q.w * p.x + q.x * p.w + q.y * p.z - q.z * p.y;
    auto j = q.w * p.y - q.x * p.z + q.y * p.w + q.z * p.x;
    auto k = q.w * p.z + q.x * p.y - q.y * p.x + q.z * p.w;

    return make_float4(i, j, k, w);
}

CUDA_LINALG_OP float4 quat_conjugate(float4 q) {
    return make_float4(-q.x, -q.y, -q.z, q.w);
}


#undef CUDA_LINALG_OP
