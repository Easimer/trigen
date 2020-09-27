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
#define SB_BENCHMARK 1
#include "s_benchmark.h"
#include "s_compute_backend.h"
#define CUDA_SUCCEEDED(HR, APICALL) ((HR = APICALL) == cudaSuccess)
#define ASSERT_CUDA_SUCCEEDED(APICALL) ((APICALL) == cudaSuccess)

#define NUMBER_OF_CLUSTERS(idx) (s.edges[(idx)].size() + 1)

// TODO(danielm): double leading underscores violate the standard
#define __hybrid__ __device__ __host__

template<typename T>
struct CUDA_Array {
    CUDA_Array() : d_buf(nullptr), N(0) {
    }

    CUDA_Array(size_t N) : d_buf(nullptr), N(N) {
        if(N != 0) {
            ASSERT_CUDA_SUCCEEDED(cudaMalloc(&d_buf, N * sizeof(T)));
        }
    }

    CUDA_Array(CUDA_Array const& other) : d_buf(nullptr), N(other.N) {
        if(!other.is_empty()) {
            ASSERT_CUDA_SUCCEEDED(cudaMalloc(&d_buf, N * sizeof(T)));
            ASSERT_CUDA_SUCCEEDED(cudaMemcpy(d_buf, other.d_buf, N * sizeof(T), cudaMemcpyDeviceToDevice));
        }
    }

    CUDA_Array(CUDA_Array&& other) : d_buf(nullptr), N(0) {
        std::swap(d_buf, other.d_buf);
        std::swap(N, other.N);
    }

    ~CUDA_Array() {
        if(d_buf != nullptr) {
            ASSERT_CUDA_SUCCEEDED(cudaFree(d_buf));
        }
    }

    CUDA_Array& operator=(CUDA_Array const& other) {
        if(!is_empty()) {
            ASSERT_CUDA_SUCCEEDED(cudaFree(d_buf));
            d_buf = nullptr;
            N = 0;
        }

        N = other.N;
        if(N != 0) {
            ASSERT_CUDA_SUCCEEDED(cudaMalloc(&d_buf, N * sizeof(T)));
            ASSERT_CUDA_SUCCEEDED(cudaMemcpy(d_buf, other.d_buf, N * sizeof(T), cudaMemcpyDeviceToDevice));
        }

        return *this;
    }

    CUDA_Array& operator=(CUDA_Array&& other) {
        if(!is_empty()) {
            ASSERT_CUDA_SUCCEEDED(cudaFree(d_buf));
            d_buf = nullptr;
            N = 0;
        }

        std::swap(d_buf, other.d_buf);
        std::swap(N, other.N);

        return *this;
    }

    bool is_empty() const {
        return d_buf == nullptr;
    }

    size_t bytes() const {
        return N * sizeof(T);
    }

    cudaError_t write_async(T const* src, cudaStream_t stream) {
        return cudaMemcpyAsync(d_buf, src, bytes(), cudaMemcpyHostToDevice, stream);
    }

    cudaError_t read_async(T* dst, cudaStream_t stream) {
        return cudaMemcpyAsync(dst, d_buf, bytes(), cudaMemcpyDeviceToHost, stream);
    }

    cudaError_t read(T* dst) {
        return cudaMemcpy(dst, d_buf, bytes(), cudaMemcpyDeviceToHost);
    }

    operator T*() {
        return d_buf;
    }

    T* d_buf;
    size_t N;
};

__hybrid__ float4 angle_axis(float a, float4 axis) {
    float s = sin(0.5f * a);

    float4 v = s * axis;
    float w = cos(0.5f * a);

    return make_float4(v.x, v.y, v.z, w);
}

__hybrid__ float4 quat_quat_mul(float4 p, float4 q) {
    float4 r;

    r.w = p.w * q.w - p.x * q.x - p.y * q.y - p.z * q.z;
    r.x = p.w * q.x + p.x * q.w + p.y * q.z - p.z * q.y;
    r.y = p.w * q.y - p.x * q.z + p.y * q.w + p.z * q.x;
    r.z = p.w * q.z + p.x * q.y - p.y * q.x + p.z * q.w;

    return r;
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

        t = normalize(quat_quat_mul(angle_axis(w, (1 / w) * omega), t));
    }

    return t;
}

struct Cluster_Matrix_Shared_Memory {
    float4 acc[4];
    float4 invRest[4];

    float4 temp[4];
    float4 diag[4];
    float4 orient[4];
};

__hybrid__ void calculate_A_i(
    float4* A_i,
    float mass,
    float4 orientation,
    float4 size,
    float4 predicted_position,
    float4 bind_pose,
    float4 center_of_mass,
    float4 bind_pose_center_of_mass,

    Cluster_Matrix_Shared_Memory* shmem
) {
    float4* temp = shmem->temp;
    float4* diag = shmem->diag;
    float4* orient = shmem->orient;
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
    float const* adjacency, unsigned N,
    float const* masses,
    float4 const* predicted_orientations,
    float4 const* sizes,
    float4 const* predicted_positions,
    float4 const* bind_pose,
    float4 const* centers_of_masses,
    float4 const* bind_pose_centers_of_masses,
    float4 const* bind_pose_inverse_bind_pose,

    Cluster_Matrix_Shared_Memory* shmem
) {
    float4* acc = shmem->acc;
    float4* invRest = shmem->invRest;

    float4 cm = centers_of_masses[i];
    float4 cm_0 = bind_pose_centers_of_masses[i];

    calculate_A_i(
            acc,
            masses[i], predicted_orientations[i], sizes[i],
            predicted_positions[i], bind_pose[i],
            cm, cm_0,
            shmem
    );

    unsigned base = i * N;
    // TODO(danielm): 2D
    for(unsigned ni = 0; ni < N; ni++) {
        float4 ntemp[4];
        float w = adjacency[base + ni];

        calculate_A_i(
            ntemp,
            masses[ni], predicted_orientations[ni], sizes[ni],
            predicted_positions[ni], bind_pose[ni],
            cm, cm_0,
            shmem
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

__global__ void k_calculate_particle_masses(unsigned N, float* d_masses, float4 const* d_sizes, float const* d_densities) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N) {
        float d_i = d_densities[i];
        float4 s_i = d_sizes[i];
        d_masses[i] = (4.0f / 3.0f) * glm::pi<float>() * s_i.x * s_i.y * s_i.z * d_i;
    }
}

__global__ void k_calculate_cluster_moment_matrices(
        float4* out, unsigned N,
        float const* adjacency,
        float const* masses,
        float4 const* predicted_orientations,
        float4 const* sizes,
        float4 const* predicted_positions,
        float4 const* bind_pose,
        float4 const* centers_of_masses,
        float4 const* bind_pose_centers_of_masses,
        float4 const* bind_pose_inverse_bind_pose) {
    unsigned id = threadIdx.x + blockDim.x * blockIdx.x;
    if(id >= N) {
        return;
    }

    extern __shared__ Cluster_Matrix_Shared_Memory shmem[];

    calculate_cluster_moment_matrix(
            &out[4 * id], id,
            adjacency, N,
            masses, predicted_orientations, sizes,
            predicted_positions, bind_pose,
            centers_of_masses, bind_pose_centers_of_masses,
            bind_pose_inverse_bind_pose,
            &shmem[threadIdx.x]
    );
}

__global__ void k_extract_rotations(
        float4* out, unsigned N,
        float4 const* A, float4 const* predicted_orientations
        ) {
    unsigned id = threadIdx.x + blockDim.x * blockIdx.x;
    if(id >= N) {
        return;
    }

    out[id] = mueller_rotation_extraction_impl(&A[id * 4], predicted_orientations[id]);
}

__global__ void k_calculate_centers_of_masses(
        float4* com, unsigned N,
        float const* adjacency,
        float const* masses,
        float4 const* predicted_position
    ) {
    unsigned id = threadIdx.x + blockDim.x * blockIdx.x;
    if(id >= N) {
        return;
    }

    float M = masses[id];
    float4 com_cur = M * predicted_position[id];
    unsigned base = id * N;
    for(unsigned ni = 0; ni < N; ni++) {
        float w = adjacency[base + ni];

        if(w > 0) {
            float m = masses[ni];
            M += m;
            com_cur += m * predicted_position[ni];
        }
    }

    com_cur.w = 0;
    com[id] = com_cur / M;
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

class Compute_CUDA : public ICompute_Backend {
public:
    Compute_CUDA(cudaStream_t stream)
    : stream(stream), current_particle_count(0) {
        printf("sb: CUDA compute backend created\n");
        // TODO(danielm): not sure if cudaEventBlockingSync would be a good idea for this event
        ASSERT_CUDA_SUCCEEDED(cudaEventCreateWithFlags(&ev_centers_of_masses_arrived, cudaEventDefault));

        compute_ref = Make_Reference_Backend();
    }

    ~Compute_CUDA() override {
        cudaEventDestroy(ev_centers_of_masses_arrived);
        cudaStreamDestroy(stream);
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
        d_out = CUDA_Array<float4>(N);

        // HACKHACKHACK: we're assuming here that s.edges[] couldn't have possibly changed if the particle count stayed constant. This is not true! 
        calculate_particle_masses(s);
        if(N != current_particle_count) {
            d_adjacency = CUDA_Array<float>(N * N);
            make_adjacency_matrix(s);
        }

        current_particle_count = N;
    }

    void calculate_particle_masses(System_State const& s) {
        auto const N = particle_count(s);

#define P_MASS_THREADS_PER_BLOCK (8)
        auto blocks = get_block_count<P_MASS_THREADS_PER_BLOCK>(N);
        cudaMemcpyAsync(d_densities, s.density.data(), d_densities.bytes(), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_sizes, s.size.data(), d_sizes.bytes(), cudaMemcpyHostToDevice, stream);
        k_calculate_particle_masses<<<blocks, P_MASS_THREADS_PER_BLOCK, 0, stream>>>(N, d_masses, d_sizes, d_densities);
    }

    void make_adjacency_matrix(System_State const& s) {
        DECLARE_BENCHMARK_BLOCK();

        BEGIN_BENCHMARK();
        auto const N = particle_count(s);
        Vector<float> h_ret;

        // TODO(danielm): are we sure that this fills the vector with 0.0f values?
        h_ret.resize(N * N);

        for(index_t i = 0; i < N; i++) {
            float* i_row = h_ret.data() + i * N;
            auto const& neighbors = s.edges.at(i);
            for(auto neighbor : neighbors) {
                i_row[neighbor] = 1.0f;
            }
        }

        cudaMemcpyAsync(d_adjacency, h_ret.data(), d_adjacency.bytes(), cudaMemcpyHostToDevice, stream);

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

        ASSERT_CUDA_SUCCEEDED(d_predicted_orientations.write_async((float4*)s.predicted_orientation.data(), stream));
        ASSERT_CUDA_SUCCEEDED(d_predicted_positions.write_async((float4*)s.predicted_position.data(), stream));
        ASSERT_CUDA_SUCCEEDED(d_bind_pose.write_async((float4*)s.bind_pose.data(), stream));
        ASSERT_CUDA_SUCCEEDED(d_bind_pose_centers_of_masses.write_async((float4*)s.bind_pose_center_of_mass.data(), stream));
        ASSERT_CUDA_SUCCEEDED(d_bind_pose_inverse_bind_pose.write_async((float4*)s.bind_pose_inverse_bind_pose.data(), stream));

        constexpr auto comcalc_threads_per_block = 32; 
        auto const comcalc_blocks = get_block_count<comcalc_threads_per_block>(N);
        k_calculate_centers_of_masses<<<comcalc_blocks, comcalc_threads_per_block, 0, stream>>>(d_centers_of_masses, N, d_adjacency, d_masses, d_predicted_positions);

        auto kernel_failed = cudaPeekAtLastError() != 0;
        if(kernel_failed) {
            printf("failed to dispatch k_calculate_centers_of_masses rc=%d\n", cudaGetLastError());
            std::terminate();
        }

        ASSERT_CUDA_SUCCEEDED(d_centers_of_masses.read_async((float4*)s.center_of_mass.data(), stream));
        ASSERT_CUDA_SUCCEEDED(cudaEventRecord(ev_centers_of_masses_arrived, stream));

        Vector<Quat> h_out;
        h_out.resize(N);

        float4* d_tmp_cluster_moment_matrices = NULL;
        ASSERT_CUDA_SUCCEEDED(cudaMalloc(&d_tmp_cluster_moment_matrices, N * 16 * sizeof(float)));

#define SHPMTCH_THREADS_PER_BLOCK (8)
        auto const blocks = get_block_count<SHPMTCH_THREADS_PER_BLOCK>(N);
        auto shared_memory_count = SHPMTCH_THREADS_PER_BLOCK * sizeof(Cluster_Matrix_Shared_Memory);

        k_calculate_cluster_moment_matrices<<<blocks, SHPMTCH_THREADS_PER_BLOCK, shared_memory_count, stream>>>(
            d_tmp_cluster_moment_matrices, N, d_adjacency, d_masses, d_predicted_orientations,
            d_sizes, d_predicted_positions, d_bind_pose, d_centers_of_masses,
            d_bind_pose_centers_of_masses, d_bind_pose_inverse_bind_pose
        );

        k_extract_rotations<<<blocks, SHPMTCH_THREADS_PER_BLOCK, 0, stream>>>(
            d_out, N,
            d_tmp_cluster_moment_matrices, d_predicted_orientations
        );
        
        // Calculate what we can on the CPU while we're waiting for the GPU

        struct Particle_Correction_Info {
            Vec4 pos_bind;
            Vec4 com_cur;
            float inv_numClusters;
        };

        Vector<Particle_Correction_Info> correction_infos;
        correction_infos.reserve(N);

        // Wait for the contents of s.center_of_mass to be available
        cudaEventSynchronize(ev_centers_of_masses_arrived);

        // Calculate what we can while we wait for the extracted quaternions
        // TODO(danielm): We could run this in a parallel cudaStream
        for (unsigned i = 0; i < N; i++) {
            auto const com0 = s.bind_pose_center_of_mass[i];
            correction_infos.push_back({});
            auto& inf = correction_infos.back();
            inf.pos_bind = s.bind_pose[i] - com0;
            inf.com_cur = s.center_of_mass[i];
            auto numClusters = NUMBER_OF_CLUSTERS(i);
            inf.inv_numClusters = 1.0f / (float)numClusters;
        }

        ASSERT_CUDA_SUCCEEDED(d_out.read_async((float4*)h_out.data(), stream));
        cudaStreamSynchronize(stream);
        cudaFree(d_tmp_cluster_moment_matrices);

        for (index_t i = 0; i < N; i++) {
            float const stiffness = 1;
            auto const R = h_out[i];

            auto& inf = correction_infos[i];

            // Rotate the bind pose position relative to the CoM
            auto pos_bind_rot = R * inf.pos_bind;
            // Our goal position
            auto goal = inf.com_cur + pos_bind_rot;
            // Number of clusters this particle is a member of
            auto correction = (goal - s.predicted_position[i]) * stiffness;
            // The correction must be divided by the number of clusters this particle is a member of
            s.predicted_position[i] += inf.inv_numClusters * correction;
            s.goal_position[i] = goal;
            s.predicted_orientation[i] = R;
        }

        END_BENCHMARK();
        PRINT_BENCHMARK_RESULT();
    }

private:
    cudaStream_t stream;

    sb::Unique_Ptr<ICompute_Backend> compute_ref;

    cudaEvent_t ev_centers_of_masses_arrived;

    size_t current_particle_count;

    CUDA_Array<float> d_adjacency;
    CUDA_Array<float> d_masses;
    CUDA_Array<float> d_densities;
    CUDA_Array<float4> d_sizes;
    CUDA_Array<float4> d_predicted_orientations;
    CUDA_Array<float4> d_centers_of_masses;
    CUDA_Array<float4> d_predicted_positions;
    CUDA_Array<float4> d_bind_pose;
    CUDA_Array<float4> d_bind_pose_centers_of_masses;
    CUDA_Array<float4> d_bind_pose_inverse_bind_pose; // mat4x4
    CUDA_Array<float4> d_out;
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
    cudaStream_t stream;

    if(init_cuda()) {
        if(CUDA_SUCCEEDED(hr, cudaStreamCreate(&stream))) {
            auto ret = std::make_unique<Compute_CUDA>(stream);
            return ret;
        } else {
            printf("sb: failed to create CUDA stream: err=%d\n", hr);
        }
    }

    fprintf(stderr, "sb: can't make CUDA compute backend\n");
    return nullptr;
}
