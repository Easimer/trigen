// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: CUDA computation backend
//

#define GLM_FORCE_CUDA
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <glm/glm.hpp>
#include "softbody.h"
#define SB_BENCHMARK 1
#include "s_benchmark.h"
#include "s_compute_backend.h"
#define CUDA_SUCCEEDED(HR, APICALL) ((HR = APICALL) == cudaSuccess)
#define ASSERT_CUDA_SUCCEEDED(APICALL) ((APICALL) == cudaSuccess)

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

    operator T*() {
        return d_buf;
    }

    T* d_buf;
    size_t N;
};

__global__ void k_calculate_particle_masses(float* d_masses, float4 const* d_sizes, float const* d_densities) {
    int i = threadIdx.x;

    float d_i = d_densities[i];
    float4 s_i = d_sizes[i];
    d_masses[i] = (4.0f / 3.0f) * M_PI * s_i.x * s_i.y * s_i.z * d_i;
}

class Compute_CUDA : public ICompute_Backend {
public:
    Compute_CUDA(cudaStream_t stream)
    : stream(stream) {
        printf("sb: CUDA compute backend created\n");
    }

    ~Compute_CUDA() override {
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
        d_masses = CUDA_Array<float>(N);
        d_densities = CUDA_Array<float>(N);
        d_sizes = CUDA_Array<float4>(N);
        // TODO: allocate buffers here

        calculate_particle_masses(s);

        d_adjacency = make_adjacency_matrix(s);
    }

    void calculate_particle_masses(System_State const& s) {
        auto N = particle_count(s);

        h_masses.resize(N);

        cudaMemcpyAsync(d_densities, s.density.data(), d_densities.bytes(), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_sizes, s.size.data(), d_sizes.bytes(), cudaMemcpyHostToDevice, stream);
        k_calculate_particle_masses<<<1, N, 0, stream>>>(d_masses, d_sizes, d_densities);
        assert(h_masses.size() * sizeof(float) == d_masses.bytes());
        cudaMemcpyAsync(h_masses.data(), d_masses, d_masses.bytes(), cudaMemcpyDeviceToHost, stream);
    }

    CUDA_Array<float> make_adjacency_matrix(System_State const& s) {
        DECLARE_BENCHMARK_BLOCK();

        BEGIN_BENCHMARK();
        auto const N = particle_count(s);
        CUDA_Array<float> d_ret(N * N);
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

        cudaMemcpyAsync(d_ret, h_ret.data(), d_ret.bytes(), cudaMemcpyHostToDevice, stream);

        END_BENCHMARK();
        PRINT_BENCHMARK_RESULT_MASKED(0xF);

        return d_ret;
    }

    void do_one_iteration_of_shape_matching_constraint_resolution(System_State& sim, float dt) override {
    }

private:
    cudaStream_t stream;

    Vector<float> h_masses;

    CUDA_Array<float> d_adjacency;
    CUDA_Array<float> d_masses;
    CUDA_Array<float> d_densities;
    CUDA_Array<float4> d_sizes;
};

static bool enumerate_devices() {
    cudaError_t hr;
    int dev_count;

    printf("CUDA version: %d\n", CUDA_VERSION);

    if(CUDA_SUCCEEDED(hr, cudaGetDeviceCount(&dev_count))) {
        cudaDeviceProp prop;
        int dev_count_ok = 0;

        for(int i = 0; i < dev_count; i++) {
            if(CUDA_SUCCEEDED(hr, cudaGetDeviceProperties(&prop, i))) {
                printf("Device #%d: '%s'\n", i, prop.name);
                dev_count_ok++;
            } else {
                printf("sb: failed to get properties of CUDA device #%d: hr=%d\n", i, hr);
            }
        }

        return dev_count_ok > 0;
    } else {
        printf("sb: failed to get CUDA device count: hr=%d\n", hr);
    }

    return false;
}

sb::Unique_Ptr<ICompute_Backend> Make_CUDA_Backend() {
    cudaError_t hr;
    cudaStream_t stream;

    if(enumerate_devices()) {
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
