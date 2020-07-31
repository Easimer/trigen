__kernel void Sum(__global float* A, __global float* output, __local float* target) {
    const size_t globalId = get_global_id(0);
    const size_t localId = get_local_id(0);
    target[localId] = A[globalId];

    barrier(CLK_LOCAL_MEM_FENCE);
    size_t blockSize = get_local_size(0);
    size_t halfBlockSize = blockSize / 2;
    while (halfBlockSize > 0) {
        if (localId < halfBlockSize) {
            target[localId] += target[localId + halfBlockSize];
            if ((halfBlockSize * 2) < blockSize) { // uneven block division
                if (localId == 0) { // when localID==0
                    target[localId] += target[localId + (blockSize - 1)];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        blockSize = halfBlockSize;
        halfBlockSize = blockSize / 2;
    }
    if (localId == 0) {
        output[get_group_id(0)] = target[0];
    }
}

void mat_add(float4* out, float4 const* lhs, float4 const* rhs) {
    out[0] = lhs[0] + rhs[0];
    out[1] = lhs[1] + rhs[1];
    out[2] = lhs[2] + rhs[2];
    out[3] = lhs[3] + rhs[3];
}

void mat_mul(
    __local float4* out,
    __local float4 const* lhs,
    __local float4 const* rhs
) {
    float4 col0 = rhs[0];
    float4 col1 = rhs[1];
    float4 col2 = rhs[2];
    float4 col3 = rhs[3];

    float4 sum0 = 0;
    float4 l0 = lhs[0];
    sum0 = fma((float4)(l0.x), col0, sum0);
    sum0 = fma((float4)(l0.y), col1, sum0);
    sum0 = fma((float4)(l0.z), col2, sum0);
    sum0 = fma((float4)(l0.w), col3, sum0);

    float4 sum1 = 0;
    float4 l1 = lhs[1];
    sum1 = fma((float4)(l1.x), col0, sum1);
    sum1 = fma((float4)(l1.y), col1, sum1);
    sum1 = fma((float4)(l1.z), col2, sum1);
    sum1 = fma((float4)(l1.w), col3, sum1);
    
    float4 sum2 = 0;
    float4 l2 = lhs[2];
    sum2 = fma((float4)(l2.x), col0, sum2);
    sum2 = fma((float4)(l2.y), col1, sum2);
    sum2 = fma((float4)(l2.z), col2, sum2);
    sum2 = fma((float4)(l2.w), col3, sum2);
        
    float4 sum3 = 0;
    float4 l3 = lhs[3];
    sum3 = fma((float4)(l3.x), col0, sum3);
    sum3 = fma((float4)(l3.y), col1, sum3);
    sum3 = fma((float4)(l3.z), col2, sum3);
    sum3 = fma((float4)(l3.w), col3, sum3);

    out[0] = sum0;
    out[1] = sum1;
    out[2] = sum2;
    out[3] = sum3;
}

__kernel
void mat_mul_main(
    __global float4* out,
    __global float4* lhs,
    __global float4* rhs,
) {
    __local float4 l_lhs[4];
    __local float4 l_rhs[4];
    __local float4 l_out[4];

    l_lhs[0] = lhs[0];
    l_lhs[1] = lhs[1];
    l_lhs[2] = lhs[2];
    l_lhs[3] = lhs[3];

    l_rhs[0] = rhs[0];
    l_rhs[1] = rhs[1];
    l_rhs[2] = rhs[2];
    l_rhs[3] = rhs[3];

    mat_mul(l_out, l_lhs, l_rhs);

    out[0] = l_out[0];
    out[1] = l_out[1];
    out[2] = l_out[2];
    out[3] = l_out[3];
}

__kernel
void calculate_particle_masses(
    __global float4* sizes,
    __global float* densities,
    __global float* masses
) {
    int i = get_global_id(0);

    float d_i = densities[i];
    float4 s_i = sizes[i];
    masses[i] = (4.0f / 3.0f) * M_PI * s_i.x * s_i.y * s_i.z * d_i;
}