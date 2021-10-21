// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: Dampening module
//

#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "math_constants.h"
#include "cuda_linalg.cuh"
#include "cuda_helper_math.h"

#include "Dampening_Scheduler.h"

__global__ void
k_dampening_velocities(
    int N,
    float4 const *pos,
    float4 const *pred_pos,
    float4 *velocity) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) {
        return;
    }

    velocity[i] = pred_pos[i] - pos[i];
}

static __global__ void
k_masses(
        int N,
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

template<typename T>
__global__ void
k_reduce(int N, T *in) {
    extern __shared__ __align__(sizeof(T)) unsigned char x_shared[];
    T *shared = reinterpret_cast<T *>(x_shared);

    auto tid = threadIdx.x;
    auto id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id > N) {
        return;
    }
    shared[tid] = in[id];
    __syncthreads();

    for (unsigned s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            shared[index] += shared[index + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        in[blockIdx.x] = shared[0];
    }
}

template<typename T>
constexpr void
reduce_shared_memory_required(
    unsigned N,
    unsigned &block_count,
    unsigned &block_size,
    unsigned &shared_mem) {
    block_size = 1024;
    shared_mem = block_size * sizeof(T);
    if (shared_mem > 32768) {
        auto excess = shared_mem - 32768 + sizeof(T);
        auto num_items = excess / sizeof(T);
        block_size -= num_items;
        shared_mem -= excess;
    }

    block_count = (N - 1) / block_size + 1;
}

namespace Dampening {

void
Scheduler::do_velocities(
        int N,
        Position_Buffer const& pos,
        Predicted_Position_Buffer const& pred_pos,
        Velocity_Buffer& velocity) {
        on_stream<Stream::Velocities>([&](cudaStream_t stream) {
            auto block_size = 1024;
            auto blocks = (N - 1) / block_size + 1;
            k_dampening_velocities<<<blocks, block_size, 0, stream>>>(
                N, pos, pred_pos, velocity);
        });
    }

    template <typename T, cudaMemcpyKind memcpyKind = cudaMemcpyDeviceToDevice>
    static void
    reduce(void *d_res, int N, CUDA_Array_Base<T> &buf, cudaStream_t stream) {
        unsigned block_size, block_count, shared_mem;
        auto remains = N;
        while (remains != 1) {
            reduce_shared_memory_required<T>(
                remains, block_count, block_size, shared_mem);
            k_reduce<T>
                <<<block_count, block_size, shared_mem, stream>>>(N, buf);
            remains = block_count;
        }

        cudaMemcpyAsync(d_res, buf, sizeof(T), memcpyKind, stream);
    }

    float
    Scheduler::do_masses(
        int N,
        Size_Buffer const &size,
        Density_Buffer const &density) {
        float ret;
        CUDA_Array<float> masses(N);
        on_stream<Stream::Masses>([&](cudaStream_t stream) {
            auto block_size = 1024;
            auto blocks = (N - 1) / block_size + 1;

            // k_masses<<<blocks, block_size, 0, stream>>>(
            //     N, masses, size, density);
            auto *asd = new float[N];
            masses.read_async(asd, stream);
            cudaStreamSynchronize(stream);
            delete[] asd;
            reduce<float, cudaMemcpyDeviceToHost>(
                &ret, N, masses.untag(), stream);
            cudaStreamSynchronize(stream);
        });
        return ret;
    }

    void
    Scheduler::do_centers_of_mass(
        int N,
        float M,
        Position_Buffer const &pos,
        Predicted_Position_Buffer const &pred_pos,
        CUDA_Array<Centers_Of_Mass> &centersOfMass) {

        on_stream<Stream::CentersOfMass>([&](cudaStream_t stream) {
            auto pos_tmp = pos.duplicate(stream);
            reduce(&centersOfMass->com0, N, pos_tmp, stream);
            auto pred_pos_tmp = pred_pos.duplicate(stream);
            reduce(&centersOfMass->com1, N, pred_pos_tmp, stream);
            Centers_Of_Mass com;
            centersOfMass.read_sub(&com, 0, 1, stream);
            cudaStreamSynchronize(stream);
            com.com0 /= M;
            com.com1 /= M;
        });
    }
    }
