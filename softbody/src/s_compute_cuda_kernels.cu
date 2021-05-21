// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: kernels for the CUDA backend
//

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "math_constants.h"

#include "cuda_linalg.cuh"
#include "cuda_helper_math.h"

#include "s_compute_cuda.h"

#define EXPLODE_F32x4(v) v.x, v.y, v.z, v.w

__device__ float4 angle_axis(float a, float4 axis) {
    float s = sin(0.5f * a);

    float4 v = s * axis;
    float w = cos(0.5f * a);

    return make_float4(v.x, v.y, v.z, w);
}

__device__ float4 angle_axis(float a, float3 axis) {
    float s = sin(0.5f * a);

    float3 v = s * axis;
    float w = cos(0.5f * a);

    return make_float4(v.x, v.y, v.z, w);
}

__device__ void dbg_print_mat4x4(float4 const* mat, char const* label, int id) {
    printf("[ %s ] #%d:\n| %f %f %f %f |\n| %f %f %f %f |\n| %f %f %f %f |\n| %f %f %f %f |\n",
            label, id,
            EXPLODE_F32x4(mat[0]), EXPLODE_F32x4(mat[1]), EXPLODE_F32x4(mat[2]), EXPLODE_F32x4(mat[3]));
}

__device__ void dbg_print_v4(float4 v, char const* label, int id) {
    printf("[ %s ] #%d:\n| %f %f %f %f |\n", label, id, EXPLODE_F32x4(v));
}

__device__ float4 mueller_rotation_extraction_impl(
    float4 const* A,
    float4 q
) {
#define MAX_ITER (16)
    float4 t = q;
    float3 const a0_xyz = xyz(A[0]);
    float3 const a1_xyz = xyz(A[1]);
    float3 const a2_xyz = xyz(A[2]);
    for(int iter = 0; iter < MAX_ITER; iter++) {
        float4 R[4];
        quat_to_mat(R, t);
        float d = 0;
        float3 c = make_float3(0, 0, 0);
        float3 r0_xyz = xyz(R[0]);
        c += cross(r0_xyz, a0_xyz);
        d += dot(r0_xyz, a0_xyz);
        float3 r1_xyz = xyz(R[1]);
        c += cross(r1_xyz, a1_xyz);
        d += dot(r1_xyz, a1_xyz);
        float3 r2_xyz = xyz(R[2]);
        c += cross(r2_xyz, a2_xyz);
        d += dot(r2_xyz, a2_xyz);
        float4 omega_v = make_float4(c, 0);
        float omega_s = 1.0f / fabs(d) + 1.0e-9;
        
        float4 omega = omega_s * omega_v;
        float w = length(omega);
        if(w < 1.0e-9) {
            break;
        }

        t = normalize(hamilton_product(angle_axis(w, (1 / w) * omega), t));
    }

    return t;
}

__device__ void calculate_A_i(
    float4* A_i,
    float mass,
    float4 orientation,
    float4 size,
    float4 predicted_position,
    float4 bind_pose,
    float4 center_of_mass,
    float4 bind_pose_center_of_mass
) {
    float4 temp[4];
    float4 diag[4];
    float4 orient[4];
    float const s = 1.0f / 5.0f;

    quat_to_mat(orient, orientation);
    diagonal3x3(diag, size * size);
    mat_mul(A_i, diag, orient);
    mat_scale(s, A_i);

    outer_product(temp, predicted_position, bind_pose);
    mat_add_assign(A_i, temp);
    outer_product(temp, center_of_mass, bind_pose_center_of_mass);
    mat_sub_assign(A_i, temp);
    mat_scale(mass, A_i);
}

__device__ void calculate_cluster_moment_matrix(
    float4* A,
    unsigned id,
    unsigned const* adjacency, unsigned adjacency_stride, unsigned N,
    float const* masses,
    float4 const* predicted_orientations,
    float4 const* sizes,
    float4 const* predicted_positions,
    float4 const* bind_pose,
    float4 const* centers_of_masses,
    float4 const* bind_pose_centers_of_masses,
    float4 const* bind_pose_inverse_bind_pose
) {
    float4 acc[4];
    float4 invRest[4];

    float4 cm = centers_of_masses[id];
    float4 cm_0 = bind_pose_centers_of_masses[id];

    calculate_A_i(
            acc,
            masses[id], predicted_orientations[id], sizes[id],
            predicted_positions[id], bind_pose[id],
            cm, cm_0
    );

    // Iterating the adjacency table
    auto adj_base = adjacency + id;
    unsigned const neighbor_count = adj_base[0];
    for(unsigned bank = 1; bank < adjacency_stride + 1; bank++) {
        float4 ntemp[4];

        // Step to the next bank
        adj_base += adjacency_stride;

        if(bank < neighbor_count + 1) {
            // ni: neighbor index
            auto ni = adj_base[0];
            calculate_A_i(
                ntemp,
                masses[ni], predicted_orientations[ni], sizes[ni],
                predicted_positions[ni], bind_pose[ni],
                cm, cm_0
            );

            acc[0] = acc[0] + ntemp[0];
            acc[1] = acc[1] + ntemp[1];
            acc[2] = acc[2] + ntemp[2];
            acc[3] = acc[3] + ntemp[3];
        } else {
            break;
        }
    }
    // END OF Iterating the adjacency table

    invRest[0] = bind_pose_inverse_bind_pose[id * 4 + 0];
    invRest[1] = bind_pose_inverse_bind_pose[id * 4 + 1];
    invRest[2] = bind_pose_inverse_bind_pose[id * 4 + 2];
    invRest[3] = bind_pose_inverse_bind_pose[id * 4 + 3];
    mat_mul(A, acc, invRest);
}

__global__ void k_calculate_particle_masses(
        unsigned N,
        float* d_masses,
        float4 const* d_sizes,
        float const* d_densities) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) {
        return;
    }

    float d_i = d_densities[i];
    float4 s_i = d_sizes[i];
    d_masses[i] = (4.0f / 3.0f) * CUDART_PI_F * s_i.x * s_i.y * s_i.z * d_i;
}

__global__ void k_calculate_cluster_moment_matrices(
        float4* out, unsigned N, unsigned offset,
        unsigned const* adjacency, unsigned adjacency_stride,
        float const* masses,
        float4 const* predicted_orientations,
        float4 const* sizes,
        float4 const* predicted_positions,
        float4 const* bind_pose,
        float4 const* centers_of_masses,
        float4 const* bind_pose_centers_of_masses,
        float4 const* bind_pose_inverse_bind_pose) {
    int id = blockDim.x * blockIdx.x + threadIdx.x + offset;
    if(id >= N) {
        return;
    }

    calculate_cluster_moment_matrix(
            &out[4 * id], id,
            adjacency, adjacency_stride, N,
            masses, predicted_orientations, sizes,
            predicted_positions, bind_pose,
            centers_of_masses, bind_pose_centers_of_masses,
            bind_pose_inverse_bind_pose
    );
}

__global__ void k_extract_rotations(
        float4* out, unsigned N, unsigned offset,
        float4 const* A, float4 const* predicted_orientations
        ) {
    int id = blockDim.x * blockIdx.x + threadIdx.x + offset;
    if(id >= N) {
        return;
    }

    // NOTE(danielm): we don't need column 4 because no rotation information is stored there
    float4 A_cache[3];
    A_cache[0] = A[id * 4 + 0];
    A_cache[1] = A[id * 4 + 1];
    A_cache[2] = A[id * 4 + 2];

    out[id] = mueller_rotation_extraction_impl(A_cache, predicted_orientations[id]);
}

__global__ void k_calculate_centers_of_masses(
        float4* com, unsigned N, unsigned offset,
        unsigned const* adjacency, unsigned adjacency_stride,
        float const* masses,
        float4 const* predicted_position
    ) {
    int id = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if(id >= N) {
        return;
    }

    float M = masses[id];
    float4 com_cur = M * predicted_position[id];

    // Iterating the adjacency table
    auto adj_base = adjacency + id;
    unsigned const neighbor_count = adj_base[0];
    for(unsigned bank = 1; bank < adjacency_stride + 1; bank++) {
        float4 ntemp[4];

        // Step to the next bank
        adj_base += adjacency_stride;

        if(bank < neighbor_count + 1) {
            // ni: neighbor index
            auto ni = adj_base[0];
            float m = masses[ni];
            M += m;
            com_cur += m * predicted_position[ni];
        } else {
            break;
        }
    }
    // END OF Iterating the adjacency table

    com_cur.w = 0;
    com[id] = com_cur / M;
}

__global__ void k_generate_correction_info(
        Particle_Correction_Info* d_info,
        unsigned const* d_adjacency, unsigned adjacency_stride, unsigned N,
        float4 const* d_bind_pose_center_of_mass,
        float4 const* d_bind_pose
        ) {
    int const id = threadIdx.x + blockDim.x * blockIdx.x;
    if(id >= N) {
        return;
    }

    float count = d_adjacency[id];

    auto const inv_num_clusters = 1.0f / count;

    auto const com0 = d_bind_pose_center_of_mass[id];
    auto const pos_bind = d_bind_pose[id] - com0;

    d_info[id] = { pos_bind, inv_num_clusters };
}

__global__ void k_apply_rotations(
        float4* d_new_predicted_positions,
        float4* d_goal_positions,
        float4 const* d_predicted_positions,
        float4 const* d_predicted_orientations,
        float4 const* d_centers_of_masses,
        float4 const* d_rotations,
        Particle_Correction_Info const* d_info,
        unsigned N, unsigned offset
        ) {
    int const id = threadIdx.x + blockDim.x * blockIdx.x + offset;
    if(id >= N) {
        return;
    }

    float const stiffness = 1;
    auto const R = d_rotations[id];
    auto const& inf = d_info[id];
    auto pos_bind_rot = hamilton_product(hamilton_product(R, inf.pos_bind), quat_conjugate(R));
    auto goal = d_centers_of_masses[id] + pos_bind_rot;
    auto pos = d_predicted_positions[id];
    auto correction = (goal - pos) * stiffness;
    d_new_predicted_positions[id] = pos + inf.inv_num_clusters * correction;
    d_goal_positions[id] = goal;
}

__global__ void k_predict(
        int N, float dt, int offset,
        float4* d_predicted_positions,
        float4* d_predicted_orientations,
        float4 const* d_positions,
        float4 const* d_orientations,
        float4 const* d_velocities,
        float4 const* d_angular_velocities,
        float const* d_masses
        ) {
    int const id = threadIdx.x + blockDim.x * blockIdx.x + offset;
    if(id >= N) {
        return;
    }

    // Velocity damping
    float3 v0_predamp = xyz(d_velocities[id]);
    float3 ang_v0_predamp = xyz(d_angular_velocities[id]);
    float d = 1 / powf(2, dt);
    float3 v0 = d * v0_predamp;
    float3 ang_v0 = d * ang_v0_predamp;

    float3 pos0 = xyz(d_positions[id]);

    float3 ext_forces = make_float3(0, -10, 0);
    float w = 1 / d_masses[id];

    float3 v = v0 + dt * w * ext_forces;
    float3 pos = pos0 + dt * v;
    float ang_v_len = length(ang_v0);

    float4 orient0 = d_orientations[id];

    float4 q;
    if(ang_v_len < 0.01f) {
        // Angular velocity is too small; for stability reasons we keep
        // the old orientation
        q = orient0;
    } else {
        // Convert angular velocity to quaternion
        float s = ang_v_len * dt / 2;
        float angle = cos(s);
        float3 axis = ang_v0 / ang_v_len * sin(s);
        q = angle_axis(angle, axis);
    }

    d_predicted_positions[id] = make_float4(pos.x, pos.y, pos.z, 0);
    d_predicted_orientations[id] = q;
}

#include "s_compute_cuda.h"

void Compute_CUDA::calculate_masses(
        int N,
        Mass_Buffer& masses,
        Size_Buffer const& sizes,
        Density_Buffer const& densities) {
    masses = Mass_Buffer(N);

    auto block_size = 1024;
    auto blocks = (N - 1) / block_size + 1;

    scheduler.insert_dependency<Stream::CopyToDev, Stream::Compute>(ev_recycler);
    scheduler.printf<Stream::Compute>("[Mass] begins\n");
    scheduler.on_stream<Stream::Compute>([&](cudaStream_t stream) {
        k_calculate_particle_masses<<<blocks, block_size, 0, stream>>>(N, masses, sizes, densities);
    });
    scheduler.printf<Stream::Compute>("[Mass] done\n");
}

void Compute_CUDA::generate_correction_info(
        int N,
        Particle_Correction_Info_Buffer& corrinfo,
        Adjacency_Table_Buffer const& adjacency,
        int adjacency_stride,
        Bind_Pose_Center_Of_Mass_Buffer const& bp_com,
        Bind_Pose_Position_Buffer const& bp_pos
        ) {
    corrinfo = Particle_Correction_Info_Buffer(N);

    scheduler.insert_dependency<Stream::CopyToDev, Stream::CorrInfo>(ev_recycler);
    scheduler.printf<Stream::CorrInfo>("[Corr] begins\n");
    scheduler.on_stream<Stream::CorrInfo>([&](cudaStream_t stream) {
        constexpr auto block_size = 32;
        auto blocks = get_block_count<block_size>(N);

        k_generate_correction_info<<<blocks, block_size, 0, stream>>>(
            corrinfo,
            adjacency, adjacency_stride, N,
            bp_com,
            bp_pos 
        );
    });
    scheduler.printf<Stream::CorrInfo>("[Corr] done\n");
}

void Compute_CUDA::apply_rotations(
        int N, int offset, int batch_size,
        New_Position_Buffer& next_pos,
        New_Goal_Position_Buffer& next_goal_pos,
        New_Rotation_Buffer const& next_orient,
        Center_Of_Mass_Buffer const& com,
        Predicted_Position_Buffer const& pred_pos, Predicted_Rotation_Buffer const& pred_rot,
        Particle_Correction_Info_Buffer const& corrinfo) {
    next_pos = New_Position_Buffer(N);
    next_goal_pos = New_Goal_Position_Buffer(N);

    auto block_size = 512;
    auto blocks = (batch_size - 1) / block_size + 1;

    scheduler.insert_dependency<Stream::CorrInfo, Stream::Compute>(ev_recycler);
    scheduler.printf<Stream::Compute>("[Apply] begins\n");
    scheduler.on_stream<Stream::Compute>([&](cudaStream_t stream) {
        k_apply_rotations<<<blocks, block_size, 0, stream>>>(
                next_pos,
                next_goal_pos,
                pred_pos,
                pred_rot,
                com,
                next_orient,
                corrinfo,
                N, offset
                );
    });
    scheduler.printf<Stream::Compute>("[Apply] done\n");
} 

void Compute_CUDA::calculate_coms(
        int N, int offset, int batch_size,
        Center_Of_Mass_Buffer& coms,
        Adjacency_Table_Buffer const& adjacency, int adjacency_stride,
        Mass_Buffer const& masses,
        Predicted_Position_Buffer const& pred_pos) {
    scheduler.printf<Stream::Compute>("[CoM] begins\n");
    scheduler.on_stream<Stream::Compute>([&](cudaStream_t stream) {
        auto block_size = 256;
        auto blocks = (batch_size - 1) / block_size + 1;
        k_calculate_centers_of_masses<<<blocks, block_size, 0, stream>>>(coms, N, offset, adjacency, adjacency_stride, masses, pred_pos);
    });
    scheduler.printf<Stream::Compute>("[CoM] done\n");
}

void Compute_CUDA::calculate_cluster_matrices(
        int N, int offset, int batch_size,
        Cluster_Matrix_Buffer& clstr_mat,
        Adjacency_Table_Buffer const& adjacency, int adjacency_stride,
        Mass_Buffer const& masses,
        Size_Buffer const& sizes,
        Predicted_Position_Buffer const& pred_pos,
        Predicted_Rotation_Buffer const& pred_rot,
        Center_Of_Mass_Buffer const& com,
        Bind_Pose_Position_Buffer const& bp_pos,
        Bind_Pose_Center_Of_Mass_Buffer const& bp_com,
        Bind_Pose_Inverse_Bind_Pose_Buffer const& bp_inv) {
    scheduler.printf<Stream::Compute>("[Clstr] begins\n");
    scheduler.on_stream<Stream::Compute>([&](cudaStream_t stream) {
        auto block_size = 256;
        auto blocks = (batch_size - 1) / block_size + 1;
        k_calculate_cluster_moment_matrices<<<blocks, block_size, 0, stream>>>(clstr_mat, N, offset, adjacency, adjacency_stride, masses, pred_rot, sizes, pred_pos, bp_pos, com, bp_com, bp_inv);
    });
    scheduler.printf<Stream::Compute>("[Clstr] done\n");
}

void Compute_CUDA::extract_rotations(
        int N, int offset, int batch_size,
        New_Rotation_Buffer& next_rotations,
        Cluster_Matrix_Buffer const& clstr_mat,
        Predicted_Rotation_Buffer const& pred_rot) {
    scheduler.printf<Stream::Compute>("[ExtRot] begins\n");
    assert(offset + batch_size <= N);
    scheduler.on_stream<Stream::Compute>([&](cudaStream_t stream) {
        auto block_size = 256;
        auto blocks = (batch_size - 1) / block_size + 1;
        k_extract_rotations<<<blocks, block_size, 0, stream>>>(next_rotations, N, offset, clstr_mat, pred_rot);
    });
    scheduler.printf<Stream::Compute>("[ExtRot] done\n");
}

void Compute_CUDA::predict(
        int N, float dt, int offset, int batch_size,
        CUDA_Array<float4>& predicted_positions,
        CUDA_Array<float4>& predicted_orientations,
        CUDA_Array<float4> const& positions,
        CUDA_Array<float4> const& orientations,
        CUDA_Array<float4> const& velocities,
        CUDA_Array<float4> const& angular_velocities,
        Mass_Buffer const& masses) {
    scheduler.printf<Stream::Predict>("[Predict] begins\n");
    scheduler.on_stream<Stream::Predict>([&](cudaStream_t stream) {
        auto const block_size = 256;
        auto blocks = (batch_size - 1) / block_size + 1;
        k_predict<<<blocks, block_size, 0, stream>>>(
            N, dt, offset,
            predicted_positions, predicted_orientations,
            positions, orientations,
            velocities, angular_velocities,
            masses
        );
    });
    scheduler.printf<Stream::Predict>("[Predict] done\n");
}
