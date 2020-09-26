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
