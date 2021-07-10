// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: CUDA computation backend
//

#include <cassert>
#include <cstdarg>
#include <array>
#include <iterator>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include "l_iterators.h"
#include "softbody.h"
#define SB_BENCHMARK 1
#define SB_BENCHMARK_UNITS microseconds
#define SB_BENCHMARK_UNITS_STR "us"
#include "s_benchmark.h"
#include "s_compute_backend.h"
#include "s_compute_cuda_codegen.h"
#include "s_compute_cuda.h"
#include <glm/gtc/type_ptr.hpp>
#include "collider_handles.h"

#include <Tracy.hpp>
#include <TracyC.h>

#include "cuda_memtrack.h"

// #define OUTPUT_SANITY_CHECK (1)
// #define ENABLE_SCHEDULER_PRINTFS (1)

#define LOG(t, l, fmt, ...) _log->log(sb::Debug_Message_Source::Compute_Backend, sb::Debug_Message_Type::t, sb::Debug_Message_Severity::l, fmt, __VA_ARGS__)

Compute_CUDA::Compute_CUDA(ILogger* logger, Raytracer &&rt, std::array<cudaStream_t, (size_t)Stream::Max>&& streams) :
    scheduler(Scheduler(std::move(streams))),
    current_particle_count(0),
    _log(logger),
    _rt(std::move(rt)) {
    LOG(Informational, Low, "cuda-backend-created", 0);

    compute_ref = Make_Reference_Backend(logger);

    sb::CUDA::memtrack::activate();
}

size_t Compute_CUDA::particle_count(System_State const& s) const {
    return s.position.size();
}

void Compute_CUDA::begin_new_frame(System_State const& s) {
    cudaError_t hr;
    auto N = particle_count(s);
    assert(N > 0);
    current_particle_count = N;

    mp_predicted_orientation = CUDA_Memory_Pin(s.predicted_orientation);
    mp_bind_pose_inverse_bind_pose = CUDA_Memory_Pin(s.bind_pose_inverse_bind_pose);
    mp_predicted_position = CUDA_Memory_Pin(s.predicted_position);
    mp_goal_position = CUDA_Memory_Pin(s.goal_position);
    mp_com0 = CUDA_Memory_Pin(s.bind_pose_center_of_mass);
    mp_com = CUDA_Memory_Pin(s.center_of_mass);
    mp_sizes = CUDA_Memory_Pin(s.size);
    mp_densities = CUDA_Memory_Pin(s.density);

    mp_position = CUDA_Memory_Pin(s.position);
    mp_orientation = CUDA_Memory_Pin(s.orientation);
    mp_velocity = CUDA_Memory_Pin(s.velocity);
    mp_angular_velocity = CUDA_Memory_Pin(s.angular_velocity);

    densities = Density_Buffer(N);
    sizes = Size_Buffer(N);
    masses = Mass_Buffer(N);

    upload_densities(N, densities, s);
    upload_sizes(N, sizes, s);
    calculate_masses(N, masses, sizes, densities);

    make_adjacency_table(N, s, d_adjacency, d_adjacency_stride);
}

void Compute_CUDA::end_frame(System_State const& sim) {
    mp_predicted_orientation = CUDA_Memory_Pin();
    mp_bind_pose_inverse_bind_pose = CUDA_Memory_Pin();
    mp_predicted_position = CUDA_Memory_Pin();
    mp_goal_position = CUDA_Memory_Pin();
    mp_com0 = CUDA_Memory_Pin();
    mp_com = CUDA_Memory_Pin();
    mp_sizes = CUDA_Memory_Pin();
    mp_densities = CUDA_Memory_Pin();
    mp_position = CUDA_Memory_Pin();
    mp_orientation = CUDA_Memory_Pin();
    mp_velocity = CUDA_Memory_Pin();
    mp_angular_velocity = CUDA_Memory_Pin();
}

void Compute_CUDA::predict(System_State& s, float dt) {
    ZoneScoped;
    DECLARE_BENCHMARK_BLOCK();
    BEGIN_BENCHMARK();

    TracyCZoneN(ctx_make_buf, "Making buffers", 1);
    auto const N = particle_count(s);
    CUDA_Array<float4> predicted_positions(N);
    CUDA_Array<float4> predicted_orientations(N);
    CUDA_Array<float4> positions(N);
    CUDA_Array<float4> orientations(N);
    CUDA_Array<float4> velocities(N);
    CUDA_Array<float4> angular_velocities(N);
    TracyCZoneEnd(ctx_make_buf);

    int const batch_size = 2048;

    auto process_batch = [&](int offset, int batch_size) {
        // Upload input subdata
        TracyCZoneN(ctx_htod, "Scheduling HtoD memcpys", 1);
        scheduler.on_stream<Stream::PredictCopyToDev>([&](cudaStream_t stream) {
            ASSERT_CUDA_SUCCEEDED(positions.write_sub((float4*)s.position.data(), offset, batch_size, stream));
            ASSERT_CUDA_SUCCEEDED(orientations.write_sub((float4*)s.orientation.data(), offset, batch_size, stream));
            ASSERT_CUDA_SUCCEEDED(velocities.write_sub((float4*)s.velocity.data(), offset, batch_size, stream));
            ASSERT_CUDA_SUCCEEDED(angular_velocities.write_sub((float4*)s.angular_velocity.data(), offset, batch_size, stream));
        });
        TracyCZoneEnd(ctx_htod);

        scheduler.insert_dependency<Stream::PredictCopyToDev, Stream::Predict>(ev_recycler);

        TracyCZoneN(ctx_kernel, "Calling kernel", 1);
        predict(N, dt, offset, batch_size,
                predicted_positions, predicted_orientations,
                positions, orientations,
                velocities, angular_velocities,
                masses);
        TracyCZoneEnd(ctx_kernel);

        scheduler.insert_dependency<Stream::Predict, Stream::CopyToHost>(ev_recycler);

        TracyCZoneN(ctx_dtoh, "Scheduling DtoH memcpys", 1);
        scheduler.on_stream<Stream::CopyToHost>([&](cudaStream_t stream) {
            ASSERT_CUDA_SUCCEEDED(predicted_positions.read_sub((float4*)s.predicted_position.data(), offset, batch_size, stream));
            ASSERT_CUDA_SUCCEEDED(predicted_orientations.read_sub((float4*)s.predicted_orientation.data(), offset, batch_size, stream));
        });
        TracyCZoneEnd(ctx_dtoh);
    };

    int remains = N;
    int offset = 0;

    while(remains >= batch_size) {
        process_batch(offset, batch_size);
        remains -= batch_size;
        offset += batch_size;
    }

    if(remains > 0) {
        process_batch(offset, remains);
    }

    TracyCZoneN(ctx_sync, "Synchronizing", 1);
    scheduler.synchronize<Stream::CopyToHost>();
    TracyCZoneEnd(ctx_sync);

    END_BENCHMARK();
    PRINT_BENCHMARK_RESULT(_log);
}

void Compute_CUDA::integrate(System_State& s, float dt) {
    compute_ref->integrate(s, dt);
}

void Compute_CUDA::dampen(System_State& s, float dt) {
    compute_ref->dampen(s, dt);
}

void Compute_CUDA::make_adjacency_table(int N, System_State const& s, Adjacency_Table_Buffer& adjacency, int& adjacency_stride) {
    ZoneScoped;
    DECLARE_BENCHMARK_BLOCK();
    BEGIN_BENCHMARK();

    // NOTE(danielm):
    // Adjacency table format:
    // +=======+=======+=======+=======+=======+=======+=======+=======+
    // |  C_1  |  C_2  |  C_3  |  C_4  | N_1_1 | N_2_1 | N_3_1 | N_4_1 |
    // +=======+=======+=======+=======+=======+=======+=======+=======+
    // | N_1_2 | N_2_2 | N_3_2 | N_4_2 | N_1_3 | N_2_3 | N_3_3 | ...   |
    // +=======+=======+=======+=======+=======+=======+=======+=======+
    // Where C_i is 'the neighbor count of particle #i' and
    // N_i_j is the 'jth neighbor of particle #i'.
    //
    // Chunks like N_1_3 - N_4_3 are called 'banks'.
    // The element count of a bank is called the stride.
    //

    unsigned max_neighbor_count = 0;
    for(index_t i = 0; i < N; i++) {
        auto c = s.edges.at(i).size();
        if(c > max_neighbor_count) {
            max_neighbor_count = c;
        }
    }

    auto const header_element_count = N;
    auto const stride = N;
    auto const table_size = header_element_count + N * stride;
    h_adjacency.maybe_realloc(table_size);
    unsigned* table = h_adjacency;

    // Make header
    for(index_t i = 0; i < N; i++) {
        auto& neighbors = s.edges.at(i);
        auto count = neighbors.size();
        table[i] = count;
    }

    unsigned* indices = table + header_element_count;

    for(index_t i = 0; i < N; i++) {
        auto& neighbors = s.edges.at(i);
        auto count = neighbors.size();

        for(int bank = 0; bank < count; bank++) {
            indices[bank * stride + i] = neighbors[bank];
        }
    }

    adjacency = Adjacency_Table_Buffer(table_size);
    adjacency_stride = stride;

    scheduler.printf<Stream::CopyToDev>("[Adjacency] begins\n");
    scheduler.on_stream<Stream::CopyToDev>([&](cudaStream_t stream) {
        ASSERT_CUDA_SUCCEEDED(adjacency.write_async(table, stream));
    });
    scheduler.printf<Stream::CopyToDev>("[Adjacency] done\n");

    scheduler.insert_dependency<Stream::CopyToDev, Stream::Compute>(ev_recycler);

    END_BENCHMARK();
    PRINT_BENCHMARK_RESULT(_log);
}

void Compute_CUDA::do_one_iteration_of_fixed_constraint_resolution(System_State& s, float phdt) {
    ZoneScoped;
    compute_ref->do_one_iteration_of_fixed_constraint_resolution(s, phdt);
}

void Compute_CUDA::do_one_iteration_of_distance_constraint_resolution(System_State& s, float phdt) {
    ZoneScoped;
    compute_ref->do_one_iteration_of_distance_constraint_resolution(s, phdt);
}

void Compute_CUDA::upload_essentials(
        int N,
        System_State const& s,
        Bind_Pose_Inverse_Bind_Pose_Buffer& bp_inv,
        Bind_Pose_Center_Of_Mass_Buffer& bp_com,
        Predicted_Rotation_Buffer& pred_rot,
        Predicted_Position_Buffer& pred_pos,
        Bind_Pose_Position_Buffer& bp_pos) {
    ZoneScoped;
    // Upload data that the kernels depend on in their entirety
    bp_inv = Bind_Pose_Inverse_Bind_Pose_Buffer(4 * N);
    bp_com = Bind_Pose_Center_Of_Mass_Buffer(N);
    pred_rot = Predicted_Rotation_Buffer(N);
    pred_pos = Predicted_Position_Buffer(N);
    bp_pos = Bind_Pose_Position_Buffer(N);

    scheduler.printf<Stream::CopyToDev>("[Essential] begins\n");
    scheduler.on_stream<Stream::CopyToDev>([&](cudaStream_t stream) {
        ASSERT_CUDA_SUCCEEDED(bp_inv.write_async((float4*)s.bind_pose_inverse_bind_pose.data(), stream));
        ASSERT_CUDA_SUCCEEDED(bp_com.write_async((float4*)s.bind_pose_center_of_mass.data(), stream));
        ASSERT_CUDA_SUCCEEDED(pred_rot.write_async((float4*)s.predicted_orientation.data(), stream));
        ASSERT_CUDA_SUCCEEDED(pred_pos.write_async((float4*)s.predicted_position.data(), stream));
        ASSERT_CUDA_SUCCEEDED(bp_pos.write_async((float4*)s.bind_pose.data(), stream));
    });
    scheduler.insert_dependency<Stream::CopyToDev, Stream::Compute>(ev_recycler);
    scheduler.printf<Stream::CopyToDev>("[Essential] done\n");
}

void Compute_CUDA::upload_sizes(
        int N,
        Size_Buffer& sizes,
        System_State const& s
        ) {
    ZoneScoped;
    sizes = Size_Buffer(N);
    scheduler.printf<Stream::CopyToDev>("[Size] begins\n");
    scheduler.on_stream<Stream::CopyToDev>([&](cudaStream_t stream) {
        ASSERT_CUDA_SUCCEEDED(sizes.write_async((float4*)s.size.data(), stream));
    });
    scheduler.printf<Stream::CopyToDev>("[Size] done\n");
}

void Compute_CUDA::upload_densities(
        int N,
        Density_Buffer& densities,
        System_State const& s
        ) {
    ZoneScoped;
    densities = Density_Buffer(N);
    scheduler.printf<Stream::CopyToDev>("[Density] begins\n");
    scheduler.on_stream<Stream::CopyToDev>([&](cudaStream_t stream) {
        ASSERT_CUDA_SUCCEEDED(densities.write_async((float*)s.density.data(), stream));
    });
    scheduler.printf<Stream::CopyToDev>("[Density] done\n");
}

void Compute_CUDA::init_coms(int N, Center_Of_Mass_Buffer& coms) {
    ZoneScoped;
    coms = Center_Of_Mass_Buffer(N);
}

void Compute_CUDA::init_cluster_matrices(int N, Cluster_Matrix_Buffer& clstr_mat) {
    ZoneScoped;
    clstr_mat = Cluster_Matrix_Buffer(4 * N);
}

void Compute_CUDA::init_rotations(int N, New_Rotation_Buffer& next_rotations) {
    ZoneScoped;
    next_rotations = New_Rotation_Buffer(N);
}

void Compute_CUDA::copy_next_state(
        int N, int offset, int batch_size,
        System_State& s,
        New_Position_Buffer const& next_pos,
        New_Rotation_Buffer const& next_rot,
        New_Goal_Position_Buffer const& next_goal_pos,
        Center_Of_Mass_Buffer const& com) {
    ZoneScoped;
    scheduler.insert_dependency<Stream::Compute, Stream::CopyToHost>(ev_recycler);
    scheduler.printf<Stream::CopyToHost>("[WriteBack] begins\n");
    scheduler.on_stream<Stream::CopyToHost>([&](cudaStream_t stream) {
        ASSERT_CUDA_SUCCEEDED(next_pos.read_sub((float4*)s.predicted_position.data(), offset, batch_size, stream));
        ASSERT_CUDA_SUCCEEDED(next_rot.read_sub((float4*)s.predicted_orientation.data(), offset, batch_size, stream));
        ASSERT_CUDA_SUCCEEDED(next_goal_pos.read_sub((float4*)s.goal_position.data(), offset, batch_size, stream));
        ASSERT_CUDA_SUCCEEDED(com.read_sub((float4*)s.center_of_mass.data(), offset, batch_size, stream));
    });
    scheduler.printf<Stream::CopyToHost>("[WriteBack] done\n");
}

void Compute_CUDA::do_one_iteration_of_shape_matching_constraint_resolution(System_State& s, float dt) {
    ZoneScoped;
    DECLARE_BENCHMARK_BLOCK();
    BEGIN_BENCHMARK();
    auto const N = particle_count(s);

    ev_recycler.flip();

    Bind_Pose_Inverse_Bind_Pose_Buffer bp_inv;
    Bind_Pose_Center_Of_Mass_Buffer bp_com;
    Predicted_Rotation_Buffer pred_rot;
    Predicted_Position_Buffer pred_pos;
    Bind_Pose_Position_Buffer bp_pos;
    Center_Of_Mass_Buffer com;
    Cluster_Matrix_Buffer clstr_mat;
    New_Rotation_Buffer next_rotations;
    New_Position_Buffer next_positions;
    New_Goal_Position_Buffer next_goal_positions;
    Particle_Correction_Info_Buffer corrinfo;

    init_coms(N, com);
    init_rotations(N, next_rotations);
    init_cluster_matrices(N, clstr_mat);

    upload_sizes(N, sizes, s);
    upload_densities(N, densities, s);
    calculate_masses(N, masses, sizes, densities);
    upload_essentials(N, s, bp_inv, bp_com, pred_rot, pred_pos, bp_pos);
    generate_correction_info(N, corrinfo, d_adjacency, d_adjacency_stride, bp_com, bp_pos);
    calculate_coms(N, 0, N, com, d_adjacency, d_adjacency_stride, masses, pred_pos);
    calculate_cluster_matrices(N, 0, N, clstr_mat, d_adjacency, d_adjacency_stride, masses, sizes, pred_pos, pred_rot, com, bp_pos, bp_com, bp_inv);
    extract_rotations(N, 0, N, next_rotations, clstr_mat, pred_rot);
    apply_rotations(N, 0, N, next_positions, next_goal_positions, next_rotations, com, pred_pos, pred_rot, corrinfo);
    copy_next_state(N, 0, N, s, next_positions, next_rotations, next_goal_positions, com);
    scheduler.synchronize<Stream::CopyToHost>();
#if OUTPUT_SANITY_CHECK
    output_sanity_check(s);
#endif /* OUTPUT_SANITY_CHECK */

    END_BENCHMARK();
    PRINT_BENCHMARK_RESULT(_log);
}

void Compute_CUDA::on_collider_added(System_State const& sim, sb::ISoftbody_Simulation::Collider_Handle handle) {
    ZoneScoped;

    Collider_Handle_Kind handle_kind;
    size_t handle_idx;
    decode_collider_handle(handle, handle_kind, handle_idx);

    switch (handle_kind) {
    case Collider_Handle_Kind::SDF:
        on_sdf_collider_added(sim, handle_idx);
        break;
    case Collider_Handle_Kind::Mesh:
        on_mesh_collider_added(sim, handle_idx);
        break;
    }
}
void Compute_CUDA::on_collider_removed(System_State const& sim, sb::ISoftbody_Simulation::Collider_Handle handle) {
    ZoneScoped;

    Collider_Handle_Kind handle_kind;
    size_t handle_idx;
    decode_collider_handle(handle, handle_kind, handle_idx);

    switch (handle_kind) {
    case Collider_Handle_Kind::SDF:
        on_sdf_collider_removed(sim, handle_idx);
        break;
    case Collider_Handle_Kind::Mesh:
        on_mesh_collider_removed(sim, handle_idx);
        break;
    }
}

void Compute_CUDA::on_collider_changed(System_State const& sim, sb::ISoftbody_Simulation::Collider_Handle handle) {
    ZoneScoped;

    Collider_Handle_Kind handle_kind;
    size_t handle_idx;
    decode_collider_handle(handle, handle_kind, handle_idx);

    switch (handle_kind) {
    case Collider_Handle_Kind::SDF:
        on_sdf_collider_changed(sim, handle_idx);
        break;
    case Collider_Handle_Kind::Mesh:
        on_mesh_collider_changed(sim, handle_idx);
        break;
    }
}

void Compute_CUDA::do_one_iteration_of_collision_constraint_resolution(System_State& s, float phdt) {
    ZoneScoped;
    DECLARE_BENCHMARK_BLOCK();
    BEGIN_BENCHMARK();

    if(coll_constraints.size() == 0) {
        return;
    }


    if(ast_kernels.size() == 0) {
        return;
    }

    auto N = particle_count(s);

    Predicted_Position_Buffer pred_pos(N);
    Position_Buffer pos(N);

    upload_sizes(N, sizes, s);
    upload_densities(N, densities, s);
    calculate_masses(N, masses, sizes, densities);

    scheduler.on_stream<Stream::CopyToDev>([&](cudaStream_t stream) {
        ASSERT_CUDA_SUCCEEDED(pred_pos.write_async((float4*)s.predicted_position.data(), stream));
        ASSERT_CUDA_SUCCEEDED(pos.write_async((float4*)s.position.data(), stream));
    });

    scheduler.insert_dependency<Stream::CopyToDev, Stream::Compute>(ev_recycler);

    // Collision resolution is the same for all programs
    // Use the first program we can find
    auto& program = *ast_kernels.begin();

    for(auto& constraint : coll_constraints) {
        scheduler.on_stream<Stream::Compute>([&](cudaStream_t stream) {
            sb::CUDA::resolve_collision_constraints(program.second, stream, N,
                    pred_pos.untag(),
                    constraint.enable, constraint.intersections, constraint.normals,
                    pos.untag(), masses.untag());
        });
    }

    scheduler.insert_dependency<Stream::Compute, Stream::CopyToHost>(ev_recycler);

    scheduler.on_stream<Stream::CopyToHost>([&](cudaStream_t stream) {
        ASSERT_CUDA_SUCCEEDED(pred_pos.read_async((float4*)s.predicted_position.data(), stream));
    });

    scheduler.synchronize<Stream::CopyToHost>();

    END_BENCHMARK();
    PRINT_BENCHMARK_RESULT(_log);
}

void Compute_CUDA::generate_collision_constraints(System_State& s) {
    ZoneScoped;
    DECLARE_BENCHMARK_BLOCK();
    BEGIN_BENCHMARK();
    auto N = particle_count(s);

    Position_Buffer pos(N);
    Predicted_Position_Buffer pred_pos(N);

    upload_sizes(N, sizes, s);
    upload_densities(N, densities, s);
    calculate_masses(N, masses, sizes, densities);

    // Upload current and predicted particle positions
    scheduler.on_stream<Stream::CopyToDev>([&](cudaStream_t stream) {
        ASSERT_CUDA_SUCCEEDED(pos.write_async((float4*)s.position.data(), stream));
        ASSERT_CUDA_SUCCEEDED(pred_pos.write_async((float4*)s.predicted_position.data(), stream));
    });

    // Coll. constr. gen. must wait for the data
    scheduler.insert_dependency<Stream::CopyToDev, Stream::Compute>(ev_recycler);

    // Clear constraints from previous simulation step
    coll_constraints.clear();

    scheduler.on_stream<Stream::Compute>([&](cudaStream_t stream) {
        for(auto& kv : ast_kernels) {
            CUDA_Array<unsigned char> enable(N);
            CUDA_Array<float3> intersections(N);
            CUDA_Array<float3> normals(N);

            sb::CUDA::generate_collision_constraints(kv.second, stream, N, enable, intersections, normals, pred_pos.untag(), pos.untag(), masses.untag());

            coll_constraints.emplace_back(Collision_Constraints { std::move(enable), std::move(intersections), std::move(normals) });
        }
    });

    END_BENCHMARK();
    PRINT_BENCHMARK_RESULT(_log);
}

void Compute_CUDA::output_sanity_check(System_State const& s) {
    cudaDeviceSynchronize();
    // === PREDICTED ORIENTATION CHECK ===
    // Orientations are unit quaternion. Quats with lengths deviating far from 1.0 are therefore degenerate.
    auto N = particle_count(s);
    for(int i = 0; i < N; i++) {
        auto q = s.predicted_orientation[i];
        auto l = length(q);
        if(glm::abs(1 - l) > glm::epsilon<float>()) {
            LOG(Error, High, "cuda-sanity-check-degenerate-orient particle-idx=%d orient=(%f %f %f %f)", i, q.x, q.y, q.z, q.w);
            assert(!"Orientation is degenerate");
        }
    }
}

void Compute_CUDA::on_sdf_collider_added(System_State const &sim, size_t idx) {
    assert(idx < sim.colliders_sdf.size());
    assert(ast_kernels.count(idx) == 0);
    auto& coll = sim.colliders_sdf[idx];
    assert(coll.used);
    auto& expr = coll.expr;

    sb::CUDA::AST_Kernel_Handle kernel_handle;
    if(sb::CUDA::compile_ast(&kernel_handle, expr)) {
        ast_kernels.emplace(std::make_pair(idx, sb::CUDA::AST_Kernel(kernel_handle)));
    } else {
        std::abort();
    }
}

void Compute_CUDA::on_sdf_collider_removed(System_State const &sim, size_t idx) {
    if(idx < sim.colliders_sdf.size()) {
        auto& coll = sim.colliders_sdf[idx];
        assert(coll.used);

        ast_kernels.erase(idx);
    } else {
        assert(!"Invalid handle");
    }
}

void Compute_CUDA::on_sdf_collider_changed(System_State const &sim, size_t idx) {
    if(idx < sim.colliders_sdf.size()) {
        auto& coll = sim.colliders_sdf[idx];
        assert(coll.used);

        if(!coll.used) {
            assert(!"Invalid handle");
            return;
        }

        // Erase old kernel
        ast_kernels.erase(idx);

        // Recompile
        auto& expr = coll.expr;
        sb::CUDA::AST_Kernel_Handle kernel_idx;
        if(sb::CUDA::compile_ast(&kernel_idx, expr)) {
            ast_kernels.emplace(std::make_pair(idx, sb::CUDA::AST_Kernel(kernel_idx)));
        } else {
            std::abort();
        }
    } else {
        assert(!"Invalid handle");
    }
}

void Compute_CUDA::on_mesh_collider_added(System_State const &sim, size_t idx) {
    if (idx < sim.colliders_mesh.size()) {
        auto &coll = sim.colliders_mesh[idx];
        if (!coll.used) {
            assert(!"Invalid handle");
            return;
        }

        rt_intersect::Mesh_Descriptor mesh = {};

        // Convert uint64_t indices to uint32_t
        std::vector<unsigned> normal_indices;
        std::vector<unsigned> vertex_indices;
        std::transform(
            coll.normal_indices.begin(),
            coll.normal_indices.end(),
            std::back_inserter(normal_indices),
            [](uint64_t x) { return (unsigned)(x & 0xFFFF'FFFF); }
        );
        std::transform(
            coll.vertex_indices.begin(),
            coll.vertex_indices.end(),
            std::back_inserter(vertex_indices),
            [](uint64_t x) { return (unsigned)(x & 0xFFFF'FFFF); }
        );

        mesh.num_triangles = coll.triangle_count;

        mesh.h_vertices = (float3 *)coll.vertices.data();
        mesh.h_vertex_indices = vertex_indices.data();
        mesh.num_vertices = vertex_indices.size();

        mesh.h_normals = (float3*)coll.normals.data();
        mesh.h_normal_indices = normal_indices.data();
        mesh.num_normals = normal_indices.size();

        mesh.transform = glm::value_ptr(coll.transform);

        auto mesh_handle = _rt->upload_mesh(&mesh);
        mesh_colliders[idx] = std::move(mesh_handle);
    } else {
        assert(!"Invalid handle");
    }
}

void Compute_CUDA::on_mesh_collider_removed(System_State const &sim, size_t idx) {
    if (idx < sim.colliders_mesh.size()) {
        auto &coll = sim.colliders_mesh[idx];
        if (!coll.used) {
            assert(!"Invalid handle");
            return;
        }

        mesh_colliders.erase(idx);
    } else {
        assert(!"Invalid handle");
    }
}

void Compute_CUDA::on_mesh_collider_changed(System_State const &sim, size_t idx) {
    if (idx < sim.colliders_mesh.size()) {
        auto &coll = sim.colliders_mesh[idx];
        if (!coll.used) {
            assert(!"Invalid handle");
            return;
        }

        _rt->notify_meshes_updated();
    } else {
        assert(!"Invalid handle");
    }
}

void
Compute_CUDA::check_intersections(
    System_State const &s,
    Vector<unsigned> &result,
    Vector<Vec3> const &from,
    Vector<Vec3> const &to) {
    compute_ref->check_intersections(s, result, from, to);
}

#undef LOG
#define LOG(t, l, fmt, ...) logger->log(sb::Debug_Message_Source::Compute_Backend, sb::Debug_Message_Type::t, sb::Debug_Message_Severity::l, fmt, __VA_ARGS__)

static int g_cudaInit = 0;
static CUdevice g_cudaDevice;
static CUcontext g_cudaContext;

static bool init_cuda(ILogger* logger) {
    assert(g_cudaInit >= 0);

    if(g_cudaInit == 0) {
        CUresult rc;
        int count;
        char dev_name[64];

        cuInit(0);

        rc = cuDeviceGetCount(&count);
        if(rc != CUDA_SUCCESS) {
            LOG(Error, High, "cuda-device-get-count-failed rc=%d", rc);
            assert(!"cuDeviceGetCount has failed");
            return false;
        }

        if(count == 0) {
            LOG(Error, High, "cuda-no-cuda-devices", 0);
            return false;
        }

        rc = cuDeviceGet(&g_cudaDevice, 0);
        if(rc != CUDA_SUCCESS) {
            LOG(Error, High, "cuda-device-get-failed rc=%d", rc);
            assert(!"cuDeviceGet has failed");
            return false;
        }

        rc = cuDeviceGetName(dev_name, 64, g_cudaDevice);
        if(rc == CUDA_SUCCESS) {
            LOG(Informational, High, "cuda-device name=\"%s\"", dev_name);
        }

        rc = cuCtxCreate(&g_cudaContext, 0, g_cudaDevice);
        if(rc != CUDA_SUCCESS) {
            LOG(Error, High, "cuda-context-create-failed rc=%d", rc);
            assert(!"cuCtxCreate has failed");
            return false;
        }

        rc = cuCtxSetCurrent(g_cudaContext);
        if(rc != CUDA_SUCCESS) {
            LOG(Error, High, "cuda-context-make-current rc=%d", rc);
            assert(!"cuCtxSetCurrent has failed");
            return false;
        }
    }

    g_cudaInit++;

    return true;
}

static void fini_cuda() {
    assert(g_cudaInit >= 0);

    if(g_cudaInit == 1) {
        cuCtxSynchronize();
        cuCtxDestroy(g_cudaContext);
    }

    g_cudaInit--;
}

sb::Unique_Ptr<ICompute_Backend> Make_CUDA_Backend(ILogger* logger) {
    // TODO(danielm): logger
    cudaError_t hr;

    if(init_cuda(logger)) {
        constexpr auto stream_n = (size_t)Stream::Max;
        std::array<cudaStream_t, stream_n> streams;
        int i;
        for(i = 0; i < stream_n; i++) {
            if(!CUDA_SUCCEEDED(hr, cudaStreamCreate(&streams[i]))) {
                break;
            }
        }

        // Did any of the stream creation calls fail?
        if(i != stream_n) {
            i--;
            while(i >= 0) {
                cudaStreamDestroy(streams[i]);
                i--;
            }

            LOG(Error, High, "cuda-stream-create-failed rc=%d", hr);
            fini_cuda();
            return nullptr;
        }

        auto rt = rt_intersect::make_instance();
        auto ret = std::make_unique<Compute_CUDA>(logger, std::move(rt), std::move(streams));
        return ret;
    }

    LOG(Error, High, "cuda-backend-creation-failed", 0);
    return nullptr;
}
