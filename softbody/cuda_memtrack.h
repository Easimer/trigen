// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: CUDA memtrack
//

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef SOFTBODY_CUDA_MEMTRACK
#undef cuMemAlloc
#undef cuMemFree
#undef cuMemAllocHost
#undef cuMemFreeHost
#endif

namespace sb::CUDA::memtrack {
    void activate();
    void flush();

    CUresult _cuMemAlloc(CUdeviceptr *dptr, size_t bytesize);
    CUresult _cuMemFree(CUdeviceptr dptr);
    CUresult _cuMemAllocHost(void **pp, size_t bytesize);
    CUresult _cuMemFreeHost(void *p);

    cudaError_t _cudaMalloc(void **devPtr, size_t size);
    cudaError_t _cudaFree(void *devPtr);
    cudaError_t _cudaMallocHost(void **ptr, size_t size);
    cudaError_t _cudaFreeHost(void *ptr);

    template<typename T>
    cudaError_t cudaMalloc(T **devPtr, size_t size) {
        return _cudaMalloc((void**)devPtr, size);
    }

    template<typename T>
    cudaError_t cudaFree(T *devPtr) {
        return _cudaFree((void*)devPtr);
    }

    template<typename T>
    cudaError_t cudaMallocHost(T **ptr, size_t size) {
        return _cudaMallocHost((void**)ptr, size);
    }

    template<typename T>
    cudaError_t cudaFreeHost(T *ptr) {
        return _cudaFreeHost((void*)ptr);
    }
}

#ifdef SOFTBODY_CUDA_MEMTRACK
#define cuMemAlloc sb::CUDA::memtrack::_cuMemAlloc
#define cuMemFree sb::CUDA::memtrack::_cuMemFree
#define cuMemAllocHost sb::CUDA::memtrack::_cuMemAllocHost
#define cuMemFreeHost sb::CUDA::memtrack::_cuMemFreeHost

#define cudaMalloc sb::CUDA::memtrack::cudaMalloc
#define cudaFree sb::CUDA::memtrack::cudaFree
#define cudaMallocHost sb::CUDA::memtrack::cudaMallocHost
#define cudaFreeHost sb::CUDA::memtrack::cudaFreeHost
#endif /* SOFTBODY_CUDA_MEMTRACK */
