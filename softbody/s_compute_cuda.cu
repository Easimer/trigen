// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: CUDA computation backend
//

#define GLM_FORCE_CUDA
#include <cassert>
#include <array>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <numeric>
#include "l_iterators.h"
#include <glm/glm.hpp>
#include "softbody.h"
#include "cuda_linalg.cuh"
#include "cuda_helper_math.h"
#include "cuda_utils.cuh"
#define SB_BENCHMARK 1
#define SB_BENCHMARK_UNITS microseconds
#define SB_BENCHMARK_UNITS_STR "us"
#include "s_benchmark.h"
#include "s_compute_backend.h"

// TODO(danielm): double leading underscores violate the standard
#define __hybrid__ __device__ __host__

__hybrid__ float4 angle_axis(float a, float4 axis) {
    float s = sin(0.5f * a);

    float4 v = s * axis;
    float w = cos(0.5f * a);

    return make_float4(v.x, v.y, v.z, w);
}

__hybrid__ float4 mueller_rotation_extraction_impl(
    float4 const* A,
    float4 q
) {
#define MAX_ITER (16)
    float4 t = q;
    for(int iter = 0; iter < MAX_ITER; iter++) {
        float4 R[4];
        quat_to_mat(R, t);
        float3 r0_xyz = xyz(R[0]);
        float3 r1_xyz = xyz(R[1]);
        float3 r2_xyz = xyz(R[2]);
        float3 a0_xyz = xyz(A[0]);
        float3 a1_xyz = xyz(A[1]);
        float3 a2_xyz = xyz(A[2]);
        float4 omega_v = make_float4(cross(r0_xyz, a0_xyz) + cross(r1_xyz, a1_xyz) + cross(r2_xyz, a2_xyz), 0);
        float omega_s = 1.0f / fabs(dot(r0_xyz, a0_xyz) + dot(r1_xyz, a1_xyz) + dot(r2_xyz, a2_xyz)) + 1.0e-9;
        
        float4 omega = omega_s * omega_v;
        float w = length(omega);
        if(w < 1.0e-9) {
            break;
        }

        t = normalize(hamilton_product(angle_axis(w, (1 / w) * omega), t));
    }

    return t;
}

__hybrid__ void calculate_A_i(
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

__hybrid__ void calculate_cluster_moment_matrix(
    float4* A,
    unsigned i,
    unsigned char const* adjacency, unsigned N,
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

    float4 cm = centers_of_masses[i];
    float4 cm_0 = bind_pose_centers_of_masses[i];

    calculate_A_i(
            acc,
            masses[i], predicted_orientations[i], sizes[i],
            predicted_positions[i], bind_pose[i],
            cm, cm_0
    );

    unsigned base = i * N;
    // TODO(danielm): 2D
    for(unsigned ni = 0; ni < N; ni++) {
        float4 ntemp[4];
        unsigned char w = adjacency[base + ni];

        calculate_A_i(
            ntemp,
            masses[ni], predicted_orientations[ni], sizes[ni],
            predicted_positions[ni], bind_pose[ni],
            cm, cm_0
        );

        acc[0] = acc[0] + w * ntemp[0];
        acc[1] = acc[1] + w * ntemp[1];
        acc[2] = acc[2] + w * ntemp[2];
        acc[3] = acc[3] + w * ntemp[3];
    }

    invRest[0] = bind_pose_inverse_bind_pose[i * 4 + 0];
    invRest[1] = bind_pose_inverse_bind_pose[i * 4 + 1];
    invRest[2] = bind_pose_inverse_bind_pose[i * 4 + 2];
    invRest[3] = bind_pose_inverse_bind_pose[i * 4 + 3];
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
    d_masses[i] = (4.0f / 3.0f) * glm::pi<float>() * s_i.x * s_i.y * s_i.z * d_i;
}

__global__ void k_calculate_cluster_moment_matrices(
        float4* out, unsigned N, unsigned offset,
        unsigned char const* adjacency,
        float const* masses,
        float4 const* predicted_orientations,
        float4 const* sizes,
        float4 const* predicted_positions,
        float4 const* bind_pose,
        float4 const* centers_of_masses,
        float4 const* bind_pose_centers_of_masses,
        float4 const* bind_pose_inverse_bind_pose) {
    unsigned id = blockDim.x * blockIdx.x + threadIdx.x + offset;
    if(id >= N) {
        return;
    }

    calculate_cluster_moment_matrix(
            &out[4 * id], id,
            adjacency, N,
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
    unsigned id = blockDim.x * blockIdx.x + threadIdx.x + offset;
    if(id >= N) {
        return;
    }

    // NOTE(danielm): we don't need column 4 because no rotation information is stored there
    float4 A_cache[3];
    A_cache[0] = A[id * 4 + 0];
    A_cache[1] = A[id * 4 + 1];
    A_cache[2] = A[id * 4 + 2];

    for(int i = 0; i < 3; i++) 
    //A_cache[3] = A[id * 4 + 3];
    out[id] = mueller_rotation_extraction_impl(A_cache, predicted_orientations[id]);
}

__global__ void k_calculate_centers_of_masses(
        float4* com, unsigned N, unsigned offset,
        unsigned char const* adjacency,
        float const* masses,
        float4 const* predicted_position
    ) {
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if(id >= N) {
        return;
    }

    float M = masses[id];
    float4 com_cur = M * predicted_position[id];
    unsigned base = id * N;
    for(unsigned ni = 0; ni < N; ni++) {
        unsigned char w = adjacency[base + ni];

        if(w != 0) {
            float m = masses[ni];
            M += m;
            com_cur += m * predicted_position[ni];
        }
    }

    com_cur.w = 0;
    com[id] = com_cur / M;
}

struct Particle_Correction_Info {
    float4 pos_bind; // bind position wrt center of mass
    float inv_num_clusters; // 1 / number of clusters
};

__global__ void k_generate_correction_info(
        Particle_Correction_Info* d_info,
        unsigned char* d_adjacency, unsigned N,
        float4* d_bind_pose_center_of_mass,
        float4* d_bind_pose
        ) {
    unsigned const id = threadIdx.x + blockDim.x * blockIdx.x;
    if(id >= N) {
        return;
    }

    auto base = id * N;
    float count = 1;
    for(unsigned ni = 0; ni < N; ni++) {
        auto w = d_adjacency[base + ni];
        if(w != 0) {
            count += 1;
        }
    }

    auto const inv_num_clusters = 1.0f / count;

    auto const com0 = d_bind_pose_center_of_mass[id];
    auto const pos_bind = d_bind_pose[id] - com0;

    d_info[id] = { pos_bind, inv_num_clusters };
}

__global__ void k_apply_rotations(
        float4* d_new_predicted_positions,
        float4* d_goal_positions,
        float4* d_predicted_positions,
        float4 const* d_predicted_orientations,
        float4 const* d_centers_of_masses,
        float4 const* d_rotations,
        Particle_Correction_Info const* d_info,
        unsigned N, unsigned offset
        ) {
    unsigned const id = threadIdx.x + blockDim.x * blockIdx.x + offset;
    if(id >= N) {
        return;
    }

    float const stiffness = 1;
    auto const R = d_rotations[id];
    auto const& inf = d_info[id];
    auto pos_bind_rot = hamilton_product(hamilton_product(R, inf.pos_bind), quat_conjugate(R));
    auto goal = d_centers_of_masses[id] + pos_bind_rot;
    auto correction = (goal - d_predicted_positions[id]) * stiffness;
    d_new_predicted_positions[id] += inf.inv_num_clusters * correction;
    d_goal_positions[id] = goal;
}

template<long threads_per_block>
static long get_block_count(long N) {
    return (N - 1) / threads_per_block + 1;
}

// TODO(danielm): decl this in some header
// Defined in s_compute_cuda_codegen.cpp
std::vector<char> generate_kernel(sb::sdf::ast::Expression<float>* expr);

class Collider_Manager {
};

enum class Stream : size_t {
    Pipeline0 = 0,
    Pipeline1,
    Pipeline2,
    Rotation_Extract,
    Rotation_Apply,
    Aux,
    Max
};

template<typename Index_Type, Index_Type N>
class CUDA_Scheduler {
public:
    CUDA_Scheduler(std::array<cudaStream_t, (size_t)N>&& streams) : _streams(streams) {}

    ~CUDA_Scheduler() {
        for(auto stream : _streams) {
            cudaStreamDestroy(stream);
        }
    }


    template<Index_Type StreamID>
    cudaError_t on_stream(std::function<cudaError_t(cudaStream_t)> const& fun) {
        static_assert((size_t)StreamID < (size_t)N, "Stream index is invalid!");

        auto stream = _streams[(size_t)StreamID];

        return fun(stream);
    }

    template<Index_Type StreamID>
    cudaError_t on_stream(std::function<void(cudaStream_t)> const& fun) {
        static_assert((size_t)StreamID < (size_t)N, "Stream index is invalid!");

        auto stream = _streams[(size_t)StreamID];

        fun(stream);

        return cudaGetLastError();
    }

    template<Index_Type GeneratorStream, Index_Type BlockedStream>
    void insert_dependency(CUDA_Event_Recycler& evr) {
        static_assert((size_t)GeneratorStream < (size_t)N, "Generator stream index is invalid!");
        static_assert((size_t)BlockedStream < (size_t)N, "Blocked stream index is invalid!");

        cudaEvent_t ev;
        evr.get(&ev);
        cudaEventRecord(ev, _streams[(size_t)GeneratorStream]);
        cudaStreamWaitEvent(_streams[(size_t)BlockedStream], ev, 0);
    }

    template<Index_Type StreamID>
    void synchronize() {
        static_assert((size_t)StreamID < (size_t)N, "Stream index is invalid!");
        cudaStreamSynchronize(_streams[(size_t)StreamID]);
    }

    template<Index_Type StreamID>
    void stall_pipeline(CUDA_Event_Recycler& evr) {
        static_assert((size_t)StreamID < (size_t)N, "Stream index is invalid!");

        cudaEvent_t ev;
        evr.get(&ev);
        cudaEventRecord(ev, _streams[(size_t)StreamID]);
        for(size_t i = 0; i < (size_t)N; i++) {
            if(i != (size_t)StreamID) {
                cudaStreamWaitEvent(_streams[i], ev, 0);
            }
        }
    }

private:
    std::array<cudaStream_t, (size_t)N> _streams;
};

class Compute_CUDA : public ICompute_Backend {
public:
    using Scheduler = CUDA_Scheduler<Stream, Stream::Max>;

    Compute_CUDA(std::array<cudaStream_t, (size_t)Stream::Max>&& streams) :
        scheduler(Scheduler(std::move(streams))),
        current_particle_count(0) {
        printf("sb: CUDA compute backend created\n");
        // TODO(danielm): not sure if cudaEventBlockingSync would be a good idea for this event

        compute_ref = Make_Reference_Backend();
    }

    size_t particle_count(System_State const& s) const {
        return s.position.size();
    }

#define SIZE_N_VEC1(N) ((N) *  1 * sizeof(float))
#define SIZE_N_VEC4(N) ((N) *  4 * sizeof(float))
#define SIZE_N_MAT4(N) ((N) * 16 * sizeof(float))

    void begin_new_frame(System_State const& s) override {
        cudaError_t hr;
        auto N = particle_count(s);
        assert(N > 0);
        d_masses = CUDA_Array<float>(N);
        d_densities = CUDA_Array<float>(N);
        d_sizes = CUDA_Array<float4>(N);
        d_predicted_orientations = CUDA_Array<float4>(N);
        d_centers_of_masses = CUDA_Array<float4>(N);
        d_predicted_positions = CUDA_Array<float4>(N);
        d_bind_pose = CUDA_Array<float4>(N);
        d_bind_pose_centers_of_masses = CUDA_Array<float4>(N);
        d_bind_pose_inverse_bind_pose = CUDA_Array<float4>(4 * N);
        d_rotations = CUDA_Array<float4>(N);
        d_tmp_cluster_moment_matrices = CUDA_Array<float4>(4 * N);
        d_goal_positions = CUDA_Array<float4>(N);
        d_corr_info = CUDA_Array<Particle_Correction_Info>(N);
        d_new_predicted_positions = CUDA_Array<float4>(N);

        // HACKHACKHACK: we're assuming here that s.edges[] couldn't have possibly changed if the particle count stayed constant. This is not true! 
        calculate_particle_masses(s);
        if(N != current_particle_count) {
            d_adjacency = CUDA_Array<unsigned char>(N * N);
            make_adjacency_matrix(s);
        }

        current_particle_count = N;
    }

    void calculate_particle_masses(System_State const& s) {
        auto const N = particle_count(s);

#define P_MASS_THREADS_PER_BLOCK (1024)
        auto blocks = get_block_count<P_MASS_THREADS_PER_BLOCK>(N);

        scheduler.on_stream<Stream::Pipeline0>([&](cudaStream_t stream) {
            cudaMemcpyAsync(d_densities, s.density.data(), d_densities.bytes(), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_sizes, s.size.data(), d_sizes.bytes(), cudaMemcpyHostToDevice, stream);
            k_calculate_particle_masses<<<blocks, P_MASS_THREADS_PER_BLOCK, 0, stream>>>(N, d_masses, d_sizes, d_densities);
        });
    }

    void make_adjacency_matrix(System_State const& s) {
        DECLARE_BENCHMARK_BLOCK();

        BEGIN_BENCHMARK();
        auto const N = particle_count(s);
        Vector<unsigned char> h_ret(N * N, 0);

        cudaHostRegister(h_ret.data(), h_ret.size(), 0);

        for(index_t i = 0; i < N; i++) {
            unsigned char* i_row = h_ret.data() + i * N;
            auto const& neighbors = s.edges.at(i);
            for(auto neighbor : neighbors) {
                i_row[neighbor] = 1;
            }
        }

        printf("sb: adjacency-matrix size=%zu\n", d_adjacency.bytes());

        scheduler.on_stream<Stream::Pipeline0>([&](cudaStream_t stream) {
            cudaMemcpyAsync(d_adjacency, h_ret.data(), d_adjacency.bytes(), cudaMemcpyHostToDevice, stream);
        });

        cudaHostUnregister(h_ret.data());

        END_BENCHMARK();
        PRINT_BENCHMARK_RESULT();
    }

    void do_one_iteration_of_fixed_constraint_resolution(System_State& s, float phdt) override {
        compute_ref->do_one_iteration_of_fixed_constraint_resolution(s, phdt);
    }

    void do_one_iteration_of_distance_constraint_resolution(System_State& s, float phdt) override {
        compute_ref->do_one_iteration_of_distance_constraint_resolution(s, phdt);
    }

    void do_one_iteration_of_shape_matching_constraint_resolution(System_State& s, float dt) override {
        DECLARE_BENCHMARK_BLOCK();
        BEGIN_BENCHMARK();
        auto const N = particle_count(s);
        Vector<float4> h_centers_of_masses(N);

        ev_recycler.flip();

        CUDA_Memory_Pin mp_predicted_orientation(s.predicted_orientation);
        CUDA_Memory_Pin mp_bind_pose_inverse_bind_pose(s.bind_pose_inverse_bind_pose);
        CUDA_Memory_Pin mp_predicted_position(s.predicted_position);
        CUDA_Memory_Pin mp_goal_position(s.goal_position);
        CUDA_Memory_Pin mp_com0(s.bind_pose_center_of_mass);
        CUDA_Memory_Pin mp_com(s.center_of_mass);

        scheduler.on_stream<Stream::Pipeline0>([&](cudaStream_t stream) {
            ASSERT_CUDA_SUCCEEDED(d_predicted_orientations.write_async((float4*)s.predicted_orientation.data(), stream));
        });
        scheduler.on_stream<Stream::Pipeline1>([&](cudaStream_t stream) {
            ASSERT_CUDA_SUCCEEDED(d_predicted_positions.write_async((float4*)s.predicted_position.data(), stream));
        });
        scheduler.on_stream<Stream::Pipeline2>([&](cudaStream_t stream) {
            ASSERT_CUDA_SUCCEEDED(d_bind_pose.write_async((float4*)s.bind_pose.data(), stream));
        });

        scheduler.stall_pipeline<Stream::Pipeline0>(ev_recycler);
        scheduler.stall_pipeline<Stream::Pipeline1>(ev_recycler);
        scheduler.stall_pipeline<Stream::Pipeline2>(ev_recycler);

        pipelined_part(N, s);

        scheduler.synchronize<Stream::Rotation_Apply>();

        END_BENCHMARK();
        PRINT_BENCHMARK_RESULT();
    }

    void pipelined_part(size_t N, System_State& s) {
        size_t particles_remain = N;
        size_t offset = 0;

        auto process_batch = [&](size_t offset, size_t batch_size) {
            scheduler.on_stream<Stream::Pipeline0>([&](cudaStream_t stream) {
                constexpr auto comcalc_threads_per_block = 256;
                auto const comcalc_blocks = get_block_count<comcalc_threads_per_block>(batch_size);
                k_calculate_centers_of_masses<<<comcalc_blocks, comcalc_threads_per_block, 0, stream>>>(d_centers_of_masses, N, offset, d_adjacency, d_masses, d_predicted_positions);

                auto kernel_failed = cudaPeekAtLastError() != 0;
                if(kernel_failed) {
                    printf("failed to dispatch k_calculate_centers_of_masses rc=%d\n", cudaGetLastError());
                    std::terminate();
                }
            });
            scheduler.on_stream<Stream::Pipeline1>([&](cudaStream_t stream) {
                ASSERT_CUDA_SUCCEEDED(d_bind_pose_centers_of_masses.write_sub((float4*)s.bind_pose_center_of_mass.data(), offset, batch_size, stream));
                ASSERT_CUDA_SUCCEEDED(d_bind_pose_inverse_bind_pose.write_sub((float4*)s.bind_pose_inverse_bind_pose.data(), 4 * offset, 4 * batch_size, stream));
            });

            // Stream Pipeline2 must wait for CoM data to be calculated and bind-pose values to arrive
            scheduler.insert_dependency<Stream::Pipeline0, Stream::Pipeline2>(ev_recycler);
            scheduler.insert_dependency<Stream::Pipeline1, Stream::Pipeline2>(ev_recycler);

            scheduler.on_stream<Stream::Pipeline2>([&](cudaStream_t stream) {
                    constexpr auto block_size = 32;
                    auto const blocks = get_block_count<block_size>(batch_size);
                    auto shared_memory_count = 0;

                    k_calculate_cluster_moment_matrices<<<blocks, block_size, shared_memory_count, stream>>>(
                            d_tmp_cluster_moment_matrices, N, offset, d_adjacency, d_masses, d_predicted_orientations,
                            d_sizes, d_predicted_positions, d_bind_pose, d_centers_of_masses,
                            d_bind_pose_centers_of_masses, d_bind_pose_inverse_bind_pose
                            );

                    auto kernel_failed = cudaPeekAtLastError() != 0;
                    if(kernel_failed) {
                        printf("failed to dispatch k_calculate_centers_of_masses rc=%d\n", cudaGetLastError());
                        std::terminate();
                    }
            });

            scheduler.insert_dependency<Stream::Pipeline2, Stream::Rotation_Extract>(ev_recycler);

            scheduler.on_stream<Stream::Rotation_Extract>([&](cudaStream_t stream) {
                constexpr auto block_size = 512;
                auto blocks = get_block_count<block_size>(batch_size);
                k_extract_rotations<<<blocks, block_size, 0, stream>>>(
                    d_rotations, N, offset,
                    d_tmp_cluster_moment_matrices, d_predicted_orientations
                );

                d_rotations.read_sub((float4*)s.predicted_orientation.data(), offset, batch_size, stream);
            });

        };

        constexpr size_t batch_size = 2048;

        while(particles_remain >= batch_size) {
            process_batch(offset, batch_size);
            particles_remain -= batch_size;
            offset += batch_size;
        }

        if(particles_remain > 0) {
            process_batch(offset, particles_remain);
        }

        scheduler.insert_dependency<Stream::Pipeline1, Stream::Aux>(ev_recycler);
        scheduler.on_stream<Stream::Aux>([&](cudaStream_t stream) {
            constexpr auto block_size = 32;
            auto blocks = get_block_count<block_size>(N);
            // TODO(danielm): we need to make sure that the adjacency matrix is present on dev by now
            k_generate_correction_info<<<blocks, block_size, 0, stream>>>(
                d_corr_info,
                d_adjacency, N,
                d_bind_pose_centers_of_masses,
                d_bind_pose
            );
        });

        scheduler.insert_dependency<Stream::Rotation_Extract, Stream::Rotation_Apply>(ev_recycler);
        scheduler.insert_dependency<Stream::Aux, Stream::Rotation_Apply>(ev_recycler);

        scheduler.on_stream<Stream::Rotation_Apply>([&](cudaStream_t stream) {
            constexpr auto block_size = 512;
            auto blocks = get_block_count<block_size>(batch_size);

            // TODO: make this one pipelineable too
            k_apply_rotations<<<blocks, block_size, 0, stream>>>(
                d_new_predicted_positions,
                d_goal_positions,
                d_predicted_positions,
                d_predicted_orientations,
                d_centers_of_masses,
                d_rotations,
                d_corr_info,
                N, offset
            );

            // d_goal_positions.read_sub((float4*)s.goal_position.data(), offset, batch_size, stream);
            d_new_predicted_positions.read_sub((float4*)s.predicted_position.data(), offset, batch_size, stream);
        });


        // Aux stream must wait for CoM data to be calculated
        scheduler.insert_dependency<Stream::Pipeline0, Stream::Aux>(ev_recycler);
        scheduler.on_stream<Stream::Aux>([&](cudaStream_t stream) {
            ASSERT_CUDA_SUCCEEDED(d_centers_of_masses.read_async((float4*)s.center_of_mass.data(), stream));
        });

        scheduler.insert_dependency<Stream::Rotation_Extract, Stream::Aux>(ev_recycler);
        scheduler.on_stream<Stream::Aux>([&](cudaStream_t stream) {
            ASSERT_CUDA_SUCCEEDED(d_rotations.read_async((float4*)s.predicted_orientation.data(), stream));
        });
    }

private:
    Scheduler scheduler;

    sb::Unique_Ptr<ICompute_Backend> compute_ref;

    CUDA_Event ev_centers_of_masses_arrived;
    CUDA_Event ev_correction_info_present;
    CUDA_Event ev_rotations_extracted;

    CUDA_Event_Recycler ev_recycler;

    size_t current_particle_count;

    CUDA_Array<unsigned char> d_adjacency;
    CUDA_Array<float> d_masses;
    CUDA_Array<float> d_densities;
    CUDA_Array<float4> d_sizes;
    CUDA_Array<float4> d_predicted_orientations;
    CUDA_Array<float4> d_centers_of_masses;
    CUDA_Array<float4> d_predicted_positions;
    CUDA_Array<float4> d_bind_pose;
    CUDA_Array<float4> d_bind_pose_centers_of_masses;
    CUDA_Array<float4> d_bind_pose_inverse_bind_pose; // mat4x4
    CUDA_Array<float4> d_rotations;
    CUDA_Array<float4> d_tmp_cluster_moment_matrices; // mat4x4
    CUDA_Array<unsigned> d_number_of_clusters;
    CUDA_Array<float4> d_new_predicted_positions;
    CUDA_Array<float4> d_goal_positions;
    CUDA_Array<Particle_Correction_Info> d_corr_info;
};

static int g_cudaInit = 0;
static CUdevice g_cudaDevice;
static CUcontext g_cudaContext;

static bool init_cuda() {
    assert(g_cudaInit >= 0);

    if(g_cudaInit == 0) {
        CUresult rc;
        int count;
        char dev_name[64];

        cuInit(0);

        rc = cuDeviceGetCount(&count);
        if(rc != CUDA_SUCCESS) {
            printf("sb: cuDeviceGetCount has failed: rc=%d\n", rc);
            assert(!"cuDeviceGetCount has failed");
            return false;
        }

        if(count == 0) {
            printf("sb: No CUDA devices were found on this host!\n");
            return false;
        }

        rc = cuDeviceGet(&g_cudaDevice, 0);
        if(rc != CUDA_SUCCESS) {
            printf("sb: cuDeviceGet has failed: rc=%d\n", rc);
            assert(!"cuDeviceGet has failed");
            return false;
        }

        rc = cuDeviceGetName(dev_name, 64, g_cudaDevice);
        if(rc == CUDA_SUCCESS) {
            printf("sb: cuda-device='%s'\n", dev_name);
        }

        rc = cuCtxCreate(&g_cudaContext, 0, g_cudaDevice);
        if(rc != CUDA_SUCCESS) {
            printf("sb: cuCtxCreate has failed: rc=%d\n", rc);
            assert(!"cuCtxCreate has failed");
            return false;
        }

        rc = cuCtxSetCurrent(g_cudaContext);
        if(rc != CUDA_SUCCESS) {
            printf("sb: cuCtxSetCurrent has failed: rc=%d\n", rc);
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

sb::Unique_Ptr<ICompute_Backend> Make_CUDA_Backend() {
    cudaError_t hr;

    if(init_cuda()) {
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

            printf("sb: failed to create CUDA stream: err=%d\n", hr);
            fini_cuda();
            return nullptr;
        }

        auto ret = std::make_unique<Compute_CUDA>(std::move(streams));
        return ret;
    }

    fprintf(stderr, "sb: can't make CUDA compute backend\n");
    return nullptr;
}
