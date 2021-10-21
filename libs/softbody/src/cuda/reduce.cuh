// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: parallel reduction
//

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_linalg.cuh"
#include "cuda_utils.cuh"

template <typename T>
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

template <typename T, typename W>
__global__ void
k_mul_by_scalar(int N, T *in, W const *weights) {
    auto id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id > N) {
        return;
    }

    in[id] = in[id] * weights[id];
}

template <typename T>
constexpr inline void
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

    if (block_count == 1 && N < block_size) {
        block_size = N;
		shared_mem = block_size * sizeof(T);
    }
}

template <typename T, cudaMemcpyKind memcpyKind = cudaMemcpyDeviceToDevice>
void
reduce(void *d_res, int N, CUDA_Array_Base<T> &buf, cudaStream_t stream) {
    unsigned block_size, block_count, shared_mem;
    auto remains = N;
    while (remains != 1) {
        reduce_shared_memory_required<T>(
            remains, block_count, block_size, shared_mem);
        k_reduce<T><<<block_count, block_size, shared_mem, stream>>>(N, buf);
        remains = block_count;
    }

    cudaMemcpyAsync(d_res, buf, sizeof(T), memcpyKind, stream);
}

template <typename T, typename W>
void
reduce_weighted(
    void *d_res,
    int N,
    CUDA_Array_Base<T> &in,
    CUDA_Array_Base<W> const &weights,
    cudaStream_t stream,
    cudaMemcpyKind memcpyKind = cudaMemcpyDeviceToDevice) {
    auto mulBlockCount = (N - 1) / 1024 + 1;
    k_mul_by_scalar<T, W><<<mulBlockCount, 1024, 0, stream>>>(N, in, weights);
	ASSERT_CUDA_SUCCEEDED(cudaGetLastError());

    unsigned block_size, block_count, shared_mem;
    auto remains = N;
    while (remains != 1) {
        reduce_shared_memory_required<T>(
            remains, block_count, block_size, shared_mem);
        k_reduce<T><<<block_count, block_size, shared_mem, stream>>>(N, in);
        ASSERT_CUDA_SUCCEEDED(cudaGetLastError());
        remains = block_count;
    }

    ASSERT_CUDA_SUCCEEDED(
        cudaMemcpyAsync(d_res, in, sizeof(T), memcpyKind, stream));
}
