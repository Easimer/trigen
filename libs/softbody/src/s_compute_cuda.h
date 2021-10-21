// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: CUDA compute backend
//

#pragma once

#include <array>
#include <rt_intersect.h>
#include "s_compute_backend.h"
#include "cuda/s_compute_cuda_codegen.h"
#include "cuda/cuda_utils.cuh"
#include <rt_intersect.h>
#include "cuda/CUDA_Scheduler.h"
#include "cuda/BufferTypes.h"
#include "cuda/Dampening_Scheduler.h"

using Raytracer = rt_intersect::Unique_Ptr<rt_intersect::IInstance>;

struct Adjacency_Table {
public:
    void maybe_realloc(size_t new_size) {
        // NOTE(danielm): we only reallocate when the requested size is either
        // greater than the current size or itsat least half as small
        auto half_size = _siz / 2;

        if(new_size > _siz || new_size <= half_size) {
            _buf = std::make_unique<unsigned[]>(new_size);
            _pin = CUDA_Memory_Pin(_buf.get(), new_size * sizeof(unsigned));
            _siz = new_size;
        }
    }

    operator unsigned*() {
        return _buf.get();
    }
private:
    std::unique_ptr<unsigned[]> _buf;
    size_t _siz = 0;
    CUDA_Memory_Pin _pin;
};

template<long threads_per_block>
static long get_block_count(long N) {
    return (N - 1) / threads_per_block + 1;
}

enum class Stream : size_t {
    CopyToDev = 0,
    CopyToHost,
    Compute,
    CorrInfo,

    Predict,
    PredictCopyToDev,
    GlobalCenterOfMass,

    Max
};

class Compute_CUDA : public ICompute_Backend {
public:
    Compute_CUDA(
        ILogger *logger,
        Raytracer &&rt,
        std::array<cudaStream_t, (size_t)Stream::Max> &&streams,
        std::array<cudaStream_t, (size_t)Dampening::Stream::Max> &&streams2
        );

private:
    using Scheduler = CUDA_Scheduler<Stream, Stream::Max>;

    size_t particle_count(System_State const& s) const;

    void begin_new_frame(System_State const& s) override;

    void end_frame(System_State const& sim) override;

    void predict(System_State& sim, float dt) override;
    void integrate(System_State& sim, float dt) override; 
    void dampen(System_State &sim, float dt) override;


    void make_adjacency_table(int N, System_State const& s, Adjacency_Table_Buffer& adjacency, int& adjacency_stride);

    void do_one_iteration_of_fixed_constraint_resolution(System_State& s, float phdt) override;

    void do_one_iteration_of_distance_constraint_resolution(System_State& s, float phdt) override;

    void upload_essentials(
        int N,
        System_State const& s,
        Bind_Pose_Inverse_Bind_Pose_Buffer& bp_inv,
        Bind_Pose_Center_Of_Mass_Buffer& bp_com,
        Predicted_Rotation_Buffer& pred_rot,
        Predicted_Position_Buffer& pred_pos,
        Bind_Pose_Position_Buffer& bp_pos);

    void upload_sizes(
        int N,
        Size_Buffer& sizes,
        System_State const& s
    );

    void upload_densities(
        int N,
        Density_Buffer& densities,
        System_State const& s
    );

    void calculate_masses(
        int N,
        Mass_Buffer& masses,
        Size_Buffer const& sizes,
        Density_Buffer const& densities);

    void generate_correction_info(
        int N,
        Particle_Correction_Info_Buffer& corrinfo,
        Adjacency_Table_Buffer const& adjacency,
        int adjacency_stride,
        Bind_Pose_Center_Of_Mass_Buffer const& bp_com,
        Bind_Pose_Position_Buffer const& bp_pos
    );

    void apply_rotations(
        int N, int offset, int batch_size,
        New_Position_Buffer& next_pos,
        New_Goal_Position_Buffer& next_goal_pos,
        New_Rotation_Buffer const& next_orient,
        Center_Of_Mass_Buffer const& com,
        Predicted_Position_Buffer const& pred_pos, Predicted_Rotation_Buffer const& pred_rot,
        Particle_Correction_Info_Buffer const& corrinfo);

    void init_coms(int N, Center_Of_Mass_Buffer& coms);

    void calculate_coms(
        int N, int offset, int batch_size,
        Center_Of_Mass_Buffer& coms,
        Adjacency_Table_Buffer const& adjacency, int adjacency_stride,
        Mass_Buffer const& masses,
        Predicted_Position_Buffer const& pred_pos);

    void init_cluster_matrices(int N, Cluster_Matrix_Buffer& clstr_mat);

    void calculate_cluster_matrices(
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
        Bind_Pose_Inverse_Bind_Pose_Buffer const& bp_inv);

    void init_rotations(int N, New_Rotation_Buffer& next_rotations);

    void extract_rotations(
        int N, int offset, int batch_size,
        New_Rotation_Buffer& next_rotations,
        Cluster_Matrix_Buffer const& clstr_mat,
        Predicted_Rotation_Buffer const& pred_rot);

    void copy_next_state(
        int N, int offset, int batch_size,
        System_State& s,
        New_Position_Buffer const& next_pos,
        New_Rotation_Buffer const& next_rot,
        New_Goal_Position_Buffer const& next_goal_pos,
        Center_Of_Mass_Buffer const& com);

    void predict(
            int N, float dt, int offset, int batch_size,
            Predicted_Position_Buffer& predicted_positions,
            Predicted_Rotation_Buffer& predicted_orientations,
            Position_Buffer const& positions,
            Rotation_Buffer const& orientations,
            Velocity_Buffer const& velocities,
            Angular_Velocity_Buffer const& angular_velocities,
			Internal_Force_Buffer const &internal_forces,
            Mass_Buffer const& masses);

    void
    compute_global_com(
        int N,
        glm::vec4 &com,
        Position_Buffer const &positions,
        Mass_Buffer const &masses);

    void
    dampen(int N, float dt, int offset, int batch_size,
        float4 global_center_of_mass,
        Predicted_Position_Buffer& predicted_positions,
        Position_Buffer& positions,
        Internal_Force_Buffer& internal_forces);

    void do_one_iteration_of_shape_matching_constraint_resolution(System_State& s, float dt) override;

    void on_collider_added(System_State const& sim, sb::ISoftbody_Simulation::Collider_Handle handle) override;

    void on_collider_removed(System_State const& sim, sb::ISoftbody_Simulation::Collider_Handle handle) override;

    void on_collider_changed(System_State const& sim, sb::ISoftbody_Simulation::Collider_Handle handle) override;

    void do_one_iteration_of_collision_constraint_resolution(System_State& s, float phdt) override;

    void generate_collision_constraints(System_State& s) override;

    void output_sanity_check(System_State const& s);

    void on_sdf_collider_added(System_State const &sim, size_t idx);
    void on_sdf_collider_removed(System_State const &sim, size_t idx);
    void on_sdf_collider_changed(System_State const &sim, size_t idx);

    void on_mesh_collider_added(System_State const &sim, size_t idx);
    void on_mesh_collider_removed(System_State const &sim, size_t idx);
    void on_mesh_collider_changed(System_State const &sim, size_t idx);

    void
    check_intersections(
        System_State const &s,
        Vector<unsigned> &result,
        Vector<Vec3> const &from,
        Vector<Vec3> const &to);


private:
    struct Collision_Constraints {
        CUDA_Array<unsigned char> enable;
        CUDA_Array<float3> intersections;
        CUDA_Array<float3> normals;
    };

    Scheduler scheduler;
    Dampening::Scheduler dampening_scheduler;

    std::vector<Collision_Constraints> coll_constraints;
    sb::Unique_Ptr<ICompute_Backend> compute_ref;

    Size_Buffer sizes;
    Density_Buffer densities;
    Mass_Buffer masses;

    CUDA_Event ev_centers_of_masses_arrived;
    CUDA_Event ev_correction_info_present;
    CUDA_Event ev_rotations_extracted;

    CUDA_Memory_Pin mp_predicted_orientation;
    CUDA_Memory_Pin mp_bind_pose_inverse_bind_pose;
    CUDA_Memory_Pin mp_predicted_position;
    CUDA_Memory_Pin mp_goal_position;
    CUDA_Memory_Pin mp_com0;
    CUDA_Memory_Pin mp_com;
    CUDA_Memory_Pin mp_sizes;
    CUDA_Memory_Pin mp_densities;
    CUDA_Memory_Pin mp_position;
    CUDA_Memory_Pin mp_orientation;
    CUDA_Memory_Pin mp_velocity;
    CUDA_Memory_Pin mp_angular_velocity;

    Adjacency_Table h_adjacency;
    Adjacency_Table_Buffer d_adjacency;
    int d_adjacency_stride;

    CUDA_Event_Recycler ev_recycler;

    size_t current_particle_count;

    // NOTE(danielm): it's important that _rt comes before mesh_colliders for
    // correct cleanup order!
    Raytracer _rt;

    // TODO(danielm): rename to ast_programs
    Map<size_t, sb::CUDA::AST_Kernel> ast_kernels;
    Map<size_t, rt_intersect::Shared_Ptr<rt_intersect::IMesh>> mesh_colliders;

    ILogger* _log;
};

