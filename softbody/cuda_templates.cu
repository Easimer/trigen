// ==================================================================
// Vector ops
// ==================================================================

__device__ float3 operator+(float3 lhs, float3 rhs) {
    return make_float3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

__device__ float3 operator*(float s, float3 v) {
    return make_float3(s * v.x, s * v.y, s * v.z);
}

__device__ float3 operator-(float3 lhs, float3 rhs) {
    return make_float3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

__device__ float dot(float3 lhs, float3 rhs) {
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

__device__ float length(float3 v) {
    return sqrt(dot(v, v));
}


// ==================================================================
// SDF primitives
// ==================================================================

__device__ float _union(float lhs, float rhs) {
    return min(lhs, rhs);
}

__device__ float _subtract(float lhs, float rhs) {
    return max(-lhs, rhs);
}

__device__ float _intersect(float lhs, float rhs) {
    return max(lhs, rhs);
}

__device__ float3 abs(float3 v) {
    return make_float3(abs(v.x), abs(v.y), abs(v.z));
}

__device__ float3 max(float3 lhs, float3 rhs) {
    return make_float3(max(lhs.x, rhs.x), max(lhs.y, rhs.y), max(lhs.z, rhs.z));
}

__device__ float _sphere(float3 sp, float radius) {
    return length(sp) - radius;
}

__device__ float _box(float3 sp, float3 size) {
    float3 q = abs(sp) - size;
    return length(max(q, make_float3(0, 0, 0))) + min(max(q.x, max(q.y, q.z)), 0.0f);
}

// ==================================================================
// Raymarching implementation and friends
// ==================================================================

__device__ float scene(float3 const _sp);

__device__ float raymarch(
        float3 origin, float3 dir,
        int steps, float epsilon
) {
    float dist = 0;
    for(int step = 0; step < steps; step++) {
        float3 p = origin + dist * dir;
        float temp = scene(p);
        if(temp < epsilon) {
            break;
        }

        dist += temp;

        if(dist > 1) {
            break;
        }
    }

    return dist;
}

__device__ float3 normalize(float3 v) {
    float3 ret;

    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);

    ret.x = v.x / len;
    ret.y = v.y / len;
    ret.z = v.z / len;

    return ret;
}

__device__ float3 find_normal(float3 const ip, float smoothness) {
    float3 ret;
    float3 xyy = make_float3(smoothness, 0, 0);
    float3 yxy = make_float3(0, smoothness, 0);
    float3 yyx = make_float3(0, 0, smoothness);

    ret.x = scene(ip + xyy) - scene(ip - xyy);
    ret.y = scene(ip + yxy) - scene(ip - yxy);
    ret.z = scene(ip + yyx) - scene(ip - yyx);

    return normalize(ret);
}

__device__ float3 xyz(float4 v) {
    return make_float3(v.x, v.y, v.z);
}

extern "C" __global__ void k_exec_sdf(
        int offset, int N,
        float4* predicted_positions,
        float4 const* positions,
        float const* masses
) {
    int id = blockDim.x * blockIdx.x + threadIdx.x + offset;
    if(id >= N) {
        return;
    }

    float3 const start = xyz(positions[id]);
    float3 thru = xyz(predicted_positions[id]);
    float3 const dir = thru - start;

    float dist = raymarch(start, dir, 32, 0.05f);

    if(0 < dist && dist < 1) {
        float3 intersect = start + dist * dir;
        float3 normal = find_normal(intersect, 1.0f);

        float3 p = thru;
        float w = 1 / masses[id];
        float3 dir2 = p - intersect;
        float d = dot(dir2, normal);
        if(d < 0) {
            float sf = d / w;
            float3 corr = -sf * w * normal;

            float3 from = p;
            float3 to = from + corr;
            predicted_positions[id] = make_float4(to.x, to.y, to.z, 0);
        }
    }
}

extern "C" __global__ void k_gen_coll_constraints(
        int offset, int N,
        unsigned char* enable,
        float3* intersections,
        float3* normals,
        float4 const* predicted_positions,
        float4 const* positions,
        float const* masses
) {
    int id = blockDim.x * blockIdx.x + threadIdx.x + offset;
    if(id >= N) {
        return;
    }

    enable[id] = 0;

    float3 const start = xyz(positions[id]);
    float3 thru = xyz(predicted_positions[id]);
    float3 const dir = thru - start;

    float dist = raymarch(start, dir, 32, 0.05f);

    if(0 < dist && dist < 1) {
        float3 intersect = start + dist * dir;
        float3 normal = find_normal(intersect, 1.0f);
        intersections[id] = intersect;
        normals[id] = normal;
        enable[id] = 1;
    }
}

extern "C" __global__ void k_resolve_coll_constraints(
        int offset, int N,
        float4* predicted_positions,
        unsigned char const* enable,
        float3 const* intersections,
        float3 const* normals,
        float4 const* positions,
        float const* masses
) {
    int id = blockDim.x * blockIdx.x + threadIdx.x + offset;
    if(id >= N) {
        return;
    }

    if(enable[id] == 0) {
        return;
    }

    float3 normal = normals[id];
    float3 intersect = intersections[id];

    float3 p = xyz(predicted_positions[id]);
    float w = 1 / masses[id];
    float3 dir2 = p - intersect;
    float d = dot(dir2, normal);
    if(d < 0) {
        float sf = d / w;
        float3 corr = -sf * w * normal;

        float3 from = p;
        float3 to = from + corr;
        predicted_positions[id] = make_float4(to.x, to.y, to.z, 0);
    }
}
