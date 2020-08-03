void mat_add(float4* out, float4 const* lhs, float4 const* rhs) {
    out[0] = lhs[0] + rhs[0];
    out[1] = lhs[1] + rhs[1];
    out[2] = lhs[2] + rhs[2];
    out[3] = lhs[3] + rhs[3];
}

void mat_add_assign(float4* out, float4 const* other) {
    out[0] += other[0];
    out[1] += other[1];
    out[2] += other[2];
    out[3] += other[3];
}

void mat_sub_assign(float4* out, float4 const* other) {
    out[0] -= other[0];
    out[1] -= other[1];
    out[2] -= other[2];
    out[3] -= other[3];
}

void mat_scale(float s, __local float* m) {
    m[0] = s * m[0];
    m[1] = s * m[1];
    m[2] = s * m[2];
    m[3] = s * m[3];
}

void mat_mul(
    float* out,
    __local float const* lhs,
    __local float const* rhs
) {
    float4 lhs_rows[4];
    for(int row = 0; row < 4; row++) {
        lhs_rows[row].x = lhs[0 * 4 + row];
        lhs_rows[row].y = lhs[1 * 4 + row];
        lhs_rows[row].z = lhs[2 * 4 + row];
        lhs_rows[row].w = lhs[3 * 4 + row];
    }

    for(int col = 0; col < 4; col++) {
        float4 rhs_col = (float4)(rhs[col * 4 + 0], rhs[col * 4 + 1], rhs[col * 4 + 2], rhs[col * 4 + 3]);
        unsigned idx = col * 4;
        for(int row = 0; row < 4; row++) {
            out[idx] = dot(lhs_rows[row], rhs_col);
            idx++;
        }
    }
}

__kernel
void mat_mul_main(
    __global float4* out,
    __global float4* lhs,
    __global float4* rhs
) {
    __local float4 l_lhs[4];
    __local float4 l_rhs[4];
    float4 l_out[4];

    l_lhs[0] = lhs[0];
    l_lhs[1] = lhs[1];
    l_lhs[2] = lhs[2];
    l_lhs[3] = lhs[3];

    l_rhs[0] = rhs[0];
    l_rhs[1] = rhs[1];
    l_rhs[2] = rhs[2];
    l_rhs[3] = rhs[3];

    barrier(CLK_LOCAL_MEM_FENCE);
    mat_mul((float*)l_out, (__local float*)l_lhs, (__local float*)l_rhs);

    out[0] = l_out[0];
    out[1] = l_out[1];
    out[2] = l_out[2];
    out[3] = l_out[3];
    barrier(CLK_LOCAL_MEM_FENCE);
}

void diagonal3x3(float4* m, float4 diag) {
    diag.w = 0;
    m[0] = (float4)(diag.xwww);
    m[1] = (float4)(diag.wyww);
    m[2] = (float4)(diag.wwzw);
    m[3] = 0;
}

void outer_product(float4* m, float4 p, float4 q) {
    m[0] = p * q.xxxx;
    m[1] = p * q.yyyy;
    m[2] = p * q.zzzz;
    m[3] = p * q.wwww;
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
    masses[i] = (4.0f / 3.0f) * M_PI_F * s_i.x * s_i.y * s_i.z * d_i;
}

#define MAX_ITER (32)
#define DECL_MAT4(name) float4 name[4]

void quat_to_mat(float4* m, float4 q) {
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
    
    m[3] = 0;
}

float4 quat_quat_mul(float4 p, float4 q) {
    float4 r;

    r.w = p.w * q.w - p.x * q.x - p.y * q.y - p.z * q.z;
    r.x = p.w * q.x + p.x * q.w + p.y * q.z - p.z * q.y;
    r.y = p.w * q.y - p.x * q.z + p.y * q.w + p.z * q.x;
    r.z = p.w * q.z + p.x * q.y - p.y * q.x + p.z * q.w;

    return r;
}

float4 angle_axis(float a, float4 axis) {
    float s = sin(0.5f * a);

    float4 v = s * axis;
    float w = cos(0.5f * a);

    return (float4)(v.xyz, w);
}

float4 mueller_rotation_extraction_impl(
    float4 const* A,
    float4 q
) {
    int i = get_global_id(0);
    float4 t = q;
    for(int iter = 0; iter < MAX_ITER; iter++) {
        DECL_MAT4(R);
        quat_to_mat(R, t);
        float4 omega_v = (float4)(cross(R[0].xyz, A[0].xyz) + cross(R[1].xyz, A[1].xyz) + cross(R[2].xyz, A[2].xyz), 0);
        float omega_s = 1.0f / fabs(dot(R[0].xyz, A[0].xyz) + dot(R[1].xyz, A[1].xyz) + dot(R[2].xyz, A[2].xyz)) + 1.0e-9;
        
        float4 omega = omega_s * omega_v;
        float w = length(omega);
        if(w < 1.0e-9) {
            break;
        }

        t = normalize(quat_quat_mul(angle_axis(w, (1 / w) * omega), t));
    }

    return t;
}

__kernel
void mueller_rotation_extraction(
    __global float4 const* A,
    __global float4* q
) {
    int i = get_global_id(0);

    float4 l_A[4];

    __global float4 const* base = A + i * 4;
    l_A[0] = base[0];
    l_A[1] = base[1];
    l_A[2] = base[2];
    l_A[3] = base[3];

    float4 l_q = q[i];

    q[i] = mueller_rotation_extraction_impl(l_A, l_q);
}

__kernel
void calculate_optimal_rotation(
    __global float4 const* A,
    __global float4 const* invRest,
    __global float4* q
) {
    int i = get_global_id(0);

    __local float4 l_A[4];
    __local float4 l_invRest[4];
    float4 l_A_pq[4];

    __global float4 const* base_A = A + i * 4;
    l_A[0] = base_A[0];
    l_A[1] = base_A[1];
    l_A[2] = base_A[2];
    l_A[3] = base_A[3];

    __global float4 const* base_invRest = invRest + i * 4;
    l_invRest[0] = base_invRest[0];
    l_invRest[1] = base_invRest[1];
    l_invRest[2] = base_invRest[2];
    l_invRest[3] = base_invRest[3];

    mat_mul((float*)l_A_pq, (__local float*)l_A, (__local float*)l_invRest);

    float4 l_q = q[i];

    q[i] = mueller_rotation_extraction_impl(l_A_pq, l_q);
}

#define IDX_MAT4_ARR(arr, i) &arr[i * 4]

void calculate_A_i(
    float4* A_i,
    unsigned i,
    __global float* masses,
    __global float4* orientations,
    __global float4* sizes,
    __global float4* predicted_positions,
    __global float4* bind_pose,
    __global float4* centers_of_masses,
    __global float4* bind_pose_centers_of_masses
) {
    float4 temp[4];
    float4 diag[4];
    float4 orient[4];
    float const s = 1.0f / 5.0f;
    float m_i = masses[i];

    quat_to_mat(orient, orientations[i]);
    diagonal3x3(diag, sizes[i] * sizes[i]);
    mat_mul(A_i, diag, orient);
    mat_scale(s, A_i);

    outer_product(temp, predicted_positions[i], bind_pose[i]);
    mat_add_assign(A_i, temp);
    mat_scale(m_i, A_i);
    outer_product(temp, centers_of_masses[i], bind_pose_centers_of_masses[i]);
    mat_sub_assign(A_i, temp);
}

void calculate_cluster_moment_matrix(
    float4* A,
    unsigned i,
    __global unsigned* adjacency, unsigned adjacency_stride,
    __global float* masses,
    __global float4* predicted_orientations,
    __global float4* sizes,
    __global float4* predicted_positions,
    __global float4* bind_pose,
    __global float4* centers_of_masses,
    __global float4* bind_pose_centers_of_masses,
    __global float4* bind_pose_inverse_bind_pose
) {
    float4 acc[4];
    calculate_A_i(acc, i, masses, predicted_orientations, sizes, predicted_positions, bind_pose, centers_of_masses, bind_pose_centers_of_masses);


    unsigned base = i * adjacency_stride;
    unsigned number_of_neighbors = adjacency[base + 0];
    for(unsigned off = 1; off < number_of_neighbors + 1; off++) {
        float4 temp[4];
        unsigned idx = adjacency[base + off];

        calculate_A_i(temp, idx, masses, predicted_orientations, sizes, predicted_positions, bind_pose, centers_of_masses, bind_pose_centers_of_masses);
        mat_add_assign(acc, temp);
    }

    mat_mul(A, acc, IDX_MAT4_ARR(bind_pose_inverse_bind_pose, i));
}

__kernel
void do_shape_matching(
    __global unsigned* adjacency, unsigned adjacency_stride,
    __global float* masses,
    __global float4* predicted_orientations,
    __global float4* sizes,
    __global float4* predicted_positions,
    __global float4* bind_pose,
    __global float4* centers_of_masses,
    __global float4* bind_pose_centers_of_masses,
    __global float4* bind_pose_inverse_bind_pose
) {
    unsigned id = get_global_id(0);
    float4 A[4];
    calculate_cluster_moment_matrix(
        A, id,
        adjacency, adjacency_stride,
        masses, predicted_orientations, sizes,
        predicted_positions, bind_pose,
        centers_of_masses, bind_pose_centers_of_masses,
        bind_pose_inverse_bind_pose
    );
    
    predicted_orientations[id] = mueller_rotation_extraction_impl(A, predicted_orientations[id]);
}