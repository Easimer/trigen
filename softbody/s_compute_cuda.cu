// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: CUDA computation backend
//

#define GLM_FORCE_CUDA
#include <cassert>
#include <cstdarg>
#include <array>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include "nvToolsExt.h"
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
#include "s_compute_cuda_codegen.h"

#include "cuda_memtrack.h"

// #define OUTPUT_SANITY_CHECK (1)
// #define ENABLE_SCHEDULER_PRINTFS (1)

// TODO(danielm): double leading underscores violate the standard
#define __hybrid__ __device__ __host__

#define EXPLODE_F32x4(v) v.x, v.y, v.z, v.w

#define LOG(t, l, fmt, ...) _log->log(sb::Debug_Message_Source::Compute_Backend, sb::Debug_Message_Type::t, sb::Debug_Message_Severity::l, fmt, __VA_ARGS__)

struct CUDA_Range {
    CUDA_Range(char const* label) {
        nvtxRangePushA(label);
    }

    ~CUDA_Range() {
        nvtxRangePop();
    }
};

__hybrid__ float4 angle_axis(float a, float4 axis) {
    float s = sin(0.5f * a);

    float4 v = s * axis;
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

__hybrid__ float4 mueller_rotation_extraction_impl(
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
    d_masses[i] = (4.0f / 3.0f) * glm::pi<float>() * s_i.x * s_i.y * s_i.z * d_i;
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

struct Particle_Correction_Info {
    float4 pos_bind; // bind position wrt center of mass
    float inv_num_clusters; // 1 / number of clusters
};

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

template<long threads_per_block>
static long get_block_count(long N) {
    return (N - 1) / threads_per_block + 1;
}

enum class Stream : size_t {
    CopyToDev = 0,
    CopyToHost,
    Compute,

    Max
};

static void cuda_cb_printf(void* user) {
    if(user != NULL) {
        auto msg = (char*)user;
        printf("%s", msg);
        delete[] msg;
    }
}

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

    template<Index_Type StreamID>
    void printf(char const* fmt, ...) {
        static_assert((size_t)StreamID < (size_t)N, "Stream index is invalid!");
#if ENABLE_SCHEDULER_PRINTFS
        va_list ap;
        va_start(ap, fmt);
        auto siz = vsnprintf(NULL, 0, fmt, ap);
        va_end(ap);
        va_start(ap, fmt);
        auto buf = new char[siz + 1];
        vsnprintf(buf, siz + 1, fmt, ap);
        buf[siz] = 0;
        va_end(ap);

        cudaLaunchHostFunc(_streams[(size_t)StreamID], cuda_cb_printf, buf);
#endif /* ENABLE_SCHEDULER_PRINTFS */
    }

private:
    std::array<cudaStream_t, (size_t)N> _streams;
};

struct Adjacency_Table {
    using Table_Type = sb::Unique_Ptr<unsigned[]>;
    Table_Type table;
    unsigned stride;
    unsigned size;

    Adjacency_Table() : table(nullptr), stride(0), size(0) {}

    Adjacency_Table(Table_Type&& table, unsigned stride, unsigned size) :
        table(std::move(table)), stride(stride), size(size) {
        assert(this->table != nullptr);

        ASSERT_CUDA_SUCCEEDED(cudaHostRegister(this->table.get(), size * sizeof(unsigned), 0));
    }

    ~Adjacency_Table() {
        if(table != nullptr) {
            ASSERT_CUDA_SUCCEEDED(cudaHostUnregister(table.get()));
        }
    }

    Adjacency_Table& operator=(Adjacency_Table&& other) {
        if(table != nullptr) {
            ASSERT_CUDA_SUCCEEDED(cudaHostUnregister(table.get()));
        }

        table = nullptr;
        stride = 0;
        size = 0;
        std::swap(table, other.table);
        std::swap(stride, other.stride);
        std::swap(size, other.size);

        return *this;
    }
};

class Compute_CUDA : public ICompute_Backend {
public:
    using Scheduler = CUDA_Scheduler<Stream, Stream::Max>;

    Compute_CUDA(ILogger* logger, std::array<cudaStream_t, (size_t)Stream::Max>&& streams) :
        scheduler(Scheduler(std::move(streams))),
        current_particle_count(0),
       _log(logger) {
        LOG(Informational, Low, "cuda-backend-created", 0);

        compute_ref = Make_Reference_Backend(logger);

        sb::CUDA::memtrack::activate();
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
        current_particle_count = N;
    }

    void make_adjacency_table(int N, System_State const& s, CUDA_Array<unsigned>& adjacency, int& adjacency_stride) {
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
        auto table = std::make_unique<unsigned[]>(table_size);

        // Make header
        for(index_t i = 0; i < N; i++) {
            auto& neighbors = s.edges.at(i);
            auto count = neighbors.size();
            table[i] = count;
        }

        unsigned* indices = table.get() + header_element_count;

        for(index_t i = 0; i < N; i++) {
            auto& neighbors = s.edges.at(i);
            auto count = neighbors.size();

            for(int bank = 0; bank < count; bank++) {
                indices[bank * stride + i] = neighbors[bank];
            }
        }

        adjacency = CUDA_Array<unsigned>(table_size);
        adjacency_stride = stride;

        scheduler.printf<Stream::CopyToDev>("[Adjacency] begins\n");
        scheduler.on_stream<Stream::CopyToDev>([&](cudaStream_t stream) {
            ASSERT_CUDA_SUCCEEDED(adjacency.write_async(table.get(), stream));
        });
        scheduler.printf<Stream::CopyToDev>("[Adjacency] done\n");

        scheduler.insert_dependency<Stream::CopyToDev, Stream::Compute>(ev_recycler);

        END_BENCHMARK();
        PRINT_BENCHMARK_RESULT();
    }

    void do_one_iteration_of_fixed_constraint_resolution(System_State& s, float phdt) override {
        compute_ref->do_one_iteration_of_fixed_constraint_resolution(s, phdt);
    }

    void do_one_iteration_of_distance_constraint_resolution(System_State& s, float phdt) override {
        compute_ref->do_one_iteration_of_distance_constraint_resolution(s, phdt);
    }

    void upload_essentials(
            int N,
            System_State const& s,
            CUDA_Array<float4>& positions,
            CUDA_Array<float4>& bp_inv,
            CUDA_Array<float4>& bp_com,
            CUDA_Array<float4>& pred_rot,
            CUDA_Array<float4>& pred_pos,
            CUDA_Array<float4>& bp_pos) {
        // Upload data that the kernels depend on in their entirety
        positions = CUDA_Array<float4>(N);
        bp_inv = CUDA_Array<float4>(4 * N);
        bp_com = CUDA_Array<float4>(N);
        pred_rot = CUDA_Array<float4>(N);
        pred_pos = CUDA_Array<float4>(N);
        bp_pos = CUDA_Array<float4>(N);

        scheduler.printf<Stream::CopyToDev>("[Essential] begins\n");
        scheduler.on_stream<Stream::CopyToDev>([&](cudaStream_t stream) {
            ASSERT_CUDA_SUCCEEDED(positions.write_async((float4*)s.position.data(), stream));
            ASSERT_CUDA_SUCCEEDED(bp_inv.write_async((float4*)s.bind_pose_inverse_bind_pose.data(), stream));
            ASSERT_CUDA_SUCCEEDED(bp_com.write_async((float4*)s.bind_pose_center_of_mass.data(), stream));
            ASSERT_CUDA_SUCCEEDED(pred_rot.write_async((float4*)s.predicted_orientation.data(), stream));
            ASSERT_CUDA_SUCCEEDED(pred_pos.write_async((float4*)s.predicted_position.data(), stream));
            ASSERT_CUDA_SUCCEEDED(bp_pos.write_async((float4*)s.bind_pose.data(), stream));
        });
        scheduler.insert_dependency<Stream::CopyToDev, Stream::Compute>(ev_recycler);
        scheduler.printf<Stream::CopyToDev>("[Essential] done\n");
    }

    void upload_sizes(
            int N,
            CUDA_Array<float4>& sizes,
            System_State const& s
            ) {
        sizes = CUDA_Array<float4>(N);
        scheduler.printf<Stream::CopyToDev>("[Size] begins\n");
        scheduler.on_stream<Stream::CopyToDev>([&](cudaStream_t stream) {
            ASSERT_CUDA_SUCCEEDED(sizes.write_async((float4*)s.size.data(), stream));
        });
        scheduler.printf<Stream::CopyToDev>("[Size] done\n");
    }

    void upload_densities(
            int N,
            CUDA_Array<float>& densities,
            System_State const& s
            ) {
        densities = CUDA_Array<float>(N);
        scheduler.printf<Stream::CopyToDev>("[Density] begins\n");
        scheduler.on_stream<Stream::CopyToDev>([&](cudaStream_t stream) {
            ASSERT_CUDA_SUCCEEDED(densities.write_async((float*)s.density.data(), stream));
        });
        scheduler.printf<Stream::CopyToDev>("[Density] done\n");
    }

    void calculate_masses(
            int N,
            CUDA_Array<float>& masses,
            CUDA_Array<float4> const& sizes,
            CUDA_Array<float> const& densities) {
        masses = CUDA_Array<float>(N);

        auto block_size = 1024;
        auto blocks = (N - 1) / block_size + 1;

        scheduler.printf<Stream::Compute>("[Mass] begins\n");
        scheduler.insert_dependency<Stream::CopyToDev, Stream::Compute>(ev_recycler);
        scheduler.on_stream<Stream::Compute>([&](cudaStream_t stream) {
            k_calculate_particle_masses<<<blocks, block_size, 0, stream>>>(N, masses, sizes, densities);
        });
        scheduler.printf<Stream::Compute>("[Mass] done\n");
    }

    void generate_correction_info(
            int N,
            CUDA_Array<Particle_Correction_Info>& corrinfo,
            CUDA_Array<unsigned> const& adjacency,
            int adjacency_stride,
            CUDA_Array<float4> const& bp_com,
            CUDA_Array<float4> const& bp_pos
            ) {
        corrinfo = CUDA_Array<Particle_Correction_Info>(N);

        scheduler.printf<Stream::Compute>("[Corr] begins\n");
        scheduler.on_stream<Stream::Compute>([&](cudaStream_t stream) {
            constexpr auto block_size = 32;
            auto blocks = get_block_count<block_size>(N);

            k_generate_correction_info<<<blocks, block_size, 0, stream>>>(
                corrinfo,
                adjacency, adjacency_stride, N,
                bp_com,
                bp_pos 
            );
        });
        scheduler.printf<Stream::Compute>("[Corr] done\n");
    }

    void apply_rotations(
            int N, int offset, int batch_size,
            CUDA_Array<float4>& next_pos,
            CUDA_Array<float4>& next_goal_pos,
            CUDA_Array<float4> const& next_orient,
            CUDA_Array<float4> const& com,
            CUDA_Array<float4> const& pred_pos, CUDA_Array<float4> const& pred_rot,
            CUDA_Array<Particle_Correction_Info> const& corrinfo) {
        next_pos = CUDA_Array<float4>(N);
        next_goal_pos = CUDA_Array<float4>(N);

        auto block_size = 512;
        auto blocks = (batch_size - 1) / block_size + 1;

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
        scheduler.insert_dependency<Stream::Compute, Stream::CopyToHost>(ev_recycler);
        scheduler.printf<Stream::Compute>("[Apply] done\n");
    } 

    void init_coms(int N, CUDA_Array<float4>& coms) {
        coms = CUDA_Array<float4>(N);
    }

    void calculate_coms(
            int N, int offset, int batch_size,
            CUDA_Array<float4>& coms,
            CUDA_Array<unsigned> adjacency, int adjacency_stride,
            CUDA_Array<float> const& masses,
            CUDA_Array<float4> const& pred_pos) {
        scheduler.printf<Stream::Compute>("[CoM] begins\n");
        scheduler.on_stream<Stream::Compute>([&](cudaStream_t stream) {
            auto block_size = 256;
            auto blocks = (batch_size - 1) / block_size + 1;
            k_calculate_centers_of_masses<<<blocks, block_size, 0, stream>>>(coms, N, offset, adjacency, adjacency_stride, masses, pred_pos);
        });
        scheduler.printf<Stream::Compute>("[CoM] done\n");
    }

    void init_cluster_matrices(int N, CUDA_Array<float4>& clstr_mat) {
        clstr_mat = CUDA_Array<float4>(4 * N);
    }

    void calculate_cluster_matrices(
            int N, int offset, int batch_size,
            CUDA_Array<float4>& clstr_mat,
            CUDA_Array<unsigned> const& adjacency, int adjacency_stride,
            CUDA_Array<float> const& masses,
            CUDA_Array<float4> const& sizes,
            CUDA_Array<float4> const& pred_pos,
            CUDA_Array<float4> const& pred_rot,
            CUDA_Array<float4> const& com,
            CUDA_Array<float4> const& bp_pos,
            CUDA_Array<float4> const& bp_com,
            CUDA_Array<float4> const& bp_inv) {
        scheduler.printf<Stream::Compute>("[Clstr] begins\n");
        scheduler.on_stream<Stream::Compute>([&](cudaStream_t stream) {
            auto block_size = 256;
            auto blocks = (batch_size - 1) / block_size + 1;
            k_calculate_cluster_moment_matrices<<<blocks, block_size, 0, stream>>>(clstr_mat, N, offset, adjacency, adjacency_stride, masses, pred_rot, sizes, pred_pos, bp_pos, com, bp_com, bp_inv);
        });
        scheduler.printf<Stream::Compute>("[Clstr] done\n");
    }

    void init_rotations(int N, CUDA_Array<float4>& next_rotations) {
        next_rotations = std::move(CUDA_Array<float4>(N));
    }

    void extract_rotations(
            int N, int offset, int batch_size,
            CUDA_Array<float4>& next_rotations,
            CUDA_Array<float4> const& clstr_mat,
            CUDA_Array<float4> const& pred_rot) {
        scheduler.printf<Stream::Compute>("[ExtRot] begins\n");
        assert(offset + batch_size <= N);
        scheduler.on_stream<Stream::Compute>([&](cudaStream_t stream) {
            auto block_size = 256;
            auto blocks = (batch_size - 1) / block_size + 1;
            k_extract_rotations<<<blocks, block_size, 0, stream>>>(next_rotations, N, offset, clstr_mat, pred_rot);
        });
        scheduler.printf<Stream::Compute>("[ExtRot] done\n");
    }

    void copy_next_state(
            int N, int offset, int batch_size,
            System_State& s,
            CUDA_Array<float4> const& next_pos,
            CUDA_Array<float4> const& next_rot,
            CUDA_Array<float4> const& next_goal_pos,
            CUDA_Array<float4> const& com) {
        scheduler.printf<Stream::CopyToHost>("[WriteBack] begins\n");
        scheduler.on_stream<Stream::CopyToHost>([&](cudaStream_t stream) {
            ASSERT_CUDA_SUCCEEDED(next_pos.read_sub((float4*)s.predicted_position.data(), offset, batch_size, stream));
            ASSERT_CUDA_SUCCEEDED(next_rot.read_sub((float4*)s.predicted_orientation.data(), offset, batch_size, stream));
            ASSERT_CUDA_SUCCEEDED(next_goal_pos.read_sub((float4*)s.goal_position.data(), offset, batch_size, stream));
            ASSERT_CUDA_SUCCEEDED(com.read_sub((float4*)s.center_of_mass.data(), offset, batch_size, stream));
        });
        scheduler.printf<Stream::CopyToHost>("[WriteBack] done\n");
    }

    void do_one_iteration_of_shape_matching_constraint_resolution(System_State& s, float dt) override {
        DECLARE_BENCHMARK_BLOCK();
        BEGIN_BENCHMARK();
        auto const N = particle_count(s);

        ev_recycler.flip();

        CUDA_Memory_Pin mp_predicted_orientation(s.predicted_orientation);
        CUDA_Memory_Pin mp_bind_pose_inverse_bind_pose(s.bind_pose_inverse_bind_pose);
        CUDA_Memory_Pin mp_predicted_position(s.predicted_position);
        CUDA_Memory_Pin mp_goal_position(s.goal_position);
        CUDA_Memory_Pin mp_com0(s.bind_pose_center_of_mass);
        CUDA_Memory_Pin mp_com(s.center_of_mass);
        CUDA_Memory_Pin mp_sizes(s.size);
        CUDA_Memory_Pin mp_densities(s.density);

        CUDA_Array<float4> positions;
        CUDA_Array<float4> bp_inv;
        CUDA_Array<float4> bp_com;
        CUDA_Array<float4> pred_rot;
        CUDA_Array<float4> pred_pos;
        CUDA_Array<float4> bp_pos;
        CUDA_Array<unsigned> adjacency;
        CUDA_Array<float4> com;
        CUDA_Array<float> masses;
        CUDA_Array<float4> sizes;
        CUDA_Array<float4> clstr_mat;
        CUDA_Array<float4> next_rotations;
        CUDA_Array<float4> next_positions;
        CUDA_Array<float4> next_goal_positions;
        CUDA_Array<float> densities;
        int adjacency_stride;
        CUDA_Array<Particle_Correction_Info> corrinfo;

        init_coms(N, com);
        init_rotations(N, next_rotations);
        init_cluster_matrices(N, clstr_mat);
        upload_essentials(N, s, positions, bp_inv, bp_com, pred_rot, pred_pos, bp_pos);

        upload_sizes(N, sizes, s);
        upload_densities(N, densities, s);
        calculate_masses(N, masses, sizes, densities);

        make_adjacency_table(N, s, adjacency, adjacency_stride);

        generate_correction_info(N, corrinfo, adjacency, adjacency_stride, bp_com, bp_pos);

        calculate_coms(N, 0, N, com, adjacency, adjacency_stride, masses, pred_pos);

        calculate_cluster_matrices(N, 0, N, clstr_mat, adjacency, adjacency_stride, masses, sizes, pred_pos, pred_rot, com, bp_pos, bp_com, bp_inv);

        extract_rotations(N, 0, N, next_rotations, clstr_mat, pred_rot);
        apply_rotations(N, 0, N, next_positions, next_goal_positions, next_rotations, com, pred_pos, pred_rot, corrinfo);

        copy_next_state(N, 0, N, s, next_positions, next_rotations, next_goal_positions,  com);
        scheduler.synchronize<Stream::CopyToHost>();
#if OUTPUT_SANITY_CHECK
        output_sanity_check(s);
#endif /* OUTPUT_SANITY_CHECK */

        END_BENCHMARK();
        PRINT_BENCHMARK_RESULT();
    }

    void on_collider_added(System_State const& sim, sb::ISoftbody_Simulation::Collider_Handle handle) override {
        assert(handle < sim.colliders_sdf.size());
        assert(ast_kernels.count(handle) == 0);
        auto& coll = sim.colliders_sdf[handle];
        assert(coll.used);
        auto& expr = coll.expr;

        sb::CUDA::AST_Kernel_Handle kernel_handle;
        if(sb::CUDA::compile_ast(&kernel_handle, expr)) {
            ast_kernels.emplace(std::make_pair(handle, sb::CUDA::AST_Kernel(kernel_handle)));
        } else {
            std::abort();
        }
    }
    void on_collider_removed(System_State const& sim, sb::ISoftbody_Simulation::Collider_Handle handle) override {
        if(handle < sim.colliders_sdf.size()) {
            auto& coll = sim.colliders_sdf[handle];
            assert(coll.used);

            ast_kernels.erase(handle);
        } else {
            assert(!"Invalid handle");
        }
    }

    void on_collider_changed(System_State const& sim, sb::ISoftbody_Simulation::Collider_Handle handle) override {
        if(handle < sim.colliders_sdf.size()) {
            auto& coll = sim.colliders_sdf[handle];
            assert(coll.used);

            if(!coll.used) {
                assert(!"Invalid handle");
                return;
            }

            // Erase old kernel
            ast_kernels.erase(handle);

            // Recompile
            auto& expr = coll.expr;
            sb::CUDA::AST_Kernel_Handle kernel_handle;
            if(sb::CUDA::compile_ast(&kernel_handle, expr)) {
                ast_kernels.emplace(std::make_pair(handle, sb::CUDA::AST_Kernel(kernel_handle)));
            } else {
                std::abort();
            }
        } else {
            assert(!"Invalid handle");
        }
    }

    void do_one_iteration_of_collision_constraint_resolution(System_State& s, float phdt) override {
        DECLARE_BENCHMARK_BLOCK();
        BEGIN_BENCHMARK();

        if(coll_constraints.size() == 0) {
            return;
        }

        if(ast_kernels.size() == 0) {
            return;
        }

        auto N = particle_count(s);
        CUDA_Memory_Pin mp_positions(s.position);
        CUDA_Memory_Pin mp_predicted_positions(s.predicted_position);

        CUDA_Array<float4> pred_pos(N);
        CUDA_Array<float4> pos(N);
        CUDA_Array<float> masses;
        CUDA_Array<float4> sizes;
        CUDA_Array<float> densities;

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
                        pred_pos,
                        constraint.enable, constraint.intersections, constraint.normals,
                        pos, masses);
            });
        }

        scheduler.insert_dependency<Stream::Compute, Stream::CopyToHost>(ev_recycler);

        scheduler.on_stream<Stream::CopyToHost>([&](cudaStream_t stream) {
            ASSERT_CUDA_SUCCEEDED(pred_pos.read_async((float4*)s.predicted_position.data(), stream));
        });

        scheduler.synchronize<Stream::CopyToHost>();

        END_BENCHMARK();
        PRINT_BENCHMARK_RESULT();
    }

    void generate_collision_constraints(System_State& s) override {
        DECLARE_BENCHMARK_BLOCK();
        BEGIN_BENCHMARK();
        auto N = particle_count(s);
        CUDA_Memory_Pin mp_positions(s.position);
        CUDA_Memory_Pin mp_predicted_positions(s.predicted_position);

        CUDA_Array<float4> sizes;
        CUDA_Array<float> densities;
        CUDA_Array<float> masses;
        CUDA_Array<float4> pos(N), pred_pos(N);

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

                sb::CUDA::generate_collision_constraints(kv.second, stream, N, enable, intersections, normals, pred_pos, pos, masses);

                coll_constraints.emplace_back(Collision_Constraints { std::move(enable), std::move(intersections), std::move(normals) });
            }
        });

        END_BENCHMARK();
        PRINT_BENCHMARK_RESULT();
    }

    void output_sanity_check(System_State const& s) {
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

private:
    struct Collision_Constraints {
        CUDA_Array<unsigned char> enable;
        CUDA_Array<float3> intersections;
        CUDA_Array<float3> normals;
    };

    Scheduler scheduler;

    std::vector<Collision_Constraints> coll_constraints;
    sb::Unique_Ptr<ICompute_Backend> compute_ref;

    CUDA_Event ev_centers_of_masses_arrived;
    CUDA_Event ev_correction_info_present;
    CUDA_Event ev_rotations_extracted;

    CUDA_Event_Recycler ev_recycler;

    size_t current_particle_count;

    // TODO(danielm): rename to ast_programs
    Map<sb::ISoftbody_Simulation::Collider_Handle, sb::CUDA::AST_Kernel> ast_kernels;

    ILogger* _log;
};

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

        auto ret = std::make_unique<Compute_CUDA>(logger, std::move(streams));
        return ret;
    }

    LOG(Error, High, "cuda-backend-creation-failed", 0);
    return nullptr;
}
