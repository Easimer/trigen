#include <cuda_runtime.h>

__device__ float _union(float lhs, float rhs) {
    return min(lhs, rhs);
}

__device__ float _subtract(float lhs, float rhs) {
    return max(-lhs, rhs);
}

__device__ float _intersect(float lhs, float rhs) {
    return max(lhs, rhs);
}

__device__ float4 operator-(float4 lhs, float4 rhs) {
    return make_float4(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, 0);
}

__device__ float4 abs(float4 v) {
    return make_float4(abs(v.x), abs(v.y), abs(v.z), 0);
}

__device__ float4 max(float4 lhs, float4 rhs) {
    return make_float4(max(lhs.x, rhs.x), max(lhs.y, rhs.y), max(lhs.z, rhs.z), 0);
}

__device__ float length(float4 v) {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ float _sphere(float4 sp, float radius) {
    return length(sp) - radius;
}

__device__ float _box(float4 sp, float4 size) {
    float4 q = abs(sp) - size;
    return length(max(q, make_float4(0, 0, 0, 0))) + min(max(q.x, max(q.y, q.z)), 0.0f);
}
