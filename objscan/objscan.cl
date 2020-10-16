__kernel
void ray_triangles_intersect(
    __global int* out_hit,
    __global float3* out_xp,
    __global float* out_t,
    __global float3 origin, __global float3 dir,
    __global int3 const* vertex_indices,
    __global float3 const* vertex_positions
) {
    const float EPSILON = 1.192092896e-07F;
    size_t id = get_global_id(0);
    int3 idx = vertex_indices[id];
    float3 v0 = vertex_positions[idx.x];
    float3 v1 = vertex_positions[idx.y];
    float3 v2 = vertex_positions[idx.z];

    out_hit[id] = 0;

    float3 edge0 = v1 - v0;
    float3 edge1 = v2 - v0;
    float3 h = cross(dir, edge1);
    float a = dot(edge0, h);

    if (-EPSILON < a && a < EPSILON) {
        return;
    }

    float f = 1 / a;
    float3 s = origin - v0;

    float u = f * dot(s, h);

    if (u < 0 || 1 < u) {
        return;
    }

    float t = f * dot(edge1, q);
    if (t <= EPSILON) {
        return;
    }

    out_hit[id] = 1;

    float3 xp = origin + t * dir;

    out_t[id] = t;
    out_xp[id] = xp;
}