// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: utility functions for the CUDA computation backend
//

#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define CUDA_SUCCEEDED(HR, APICALL) ((HR = APICALL) == cudaSuccess)
#ifdef NDEBUG
#define ASSERT_CUDA_SUCCEEDED(APICALL) ((APICALL) == cudaSuccess)
#else
#define ASSERT_CUDA_SUCCEEDED(APICALL) assert((APICALL) == cudaSuccess)
#endif

class CUDA_Event {
public:
    CUDA_Event(unsigned flags) {
        ASSERT_CUDA_SUCCEEDED(cudaEventCreateWithFlags(&_ev, flags));
    }

    CUDA_Event() : CUDA_Event(cudaEventDefault) {}

    ~CUDA_Event() {
        cudaEventDestroy(_ev);
    }

    operator cudaEvent_t() const { return _ev; }
private:
    cudaEvent_t _ev;
};

// RAII wrapper for memory pinning
struct CUDA_Memory_Pin {
public:
    CUDA_Memory_Pin(void* ptr, size_t size) : _ptr(ptr) {
        cudaHostRegister(ptr, size, cudaHostRegisterDefault);
    }

    template<typename T>
    CUDA_Memory_Pin(std::vector<T>& v) : _ptr(v.data()) {
        // NOTE(danielm): you must be careful not to push items into the vector
        // since a vector grow might move the memory away from the current address.
        auto size = v.size() * sizeof(T);
        cudaHostRegister(_ptr, size, cudaHostRegisterDefault);
    }

    ~CUDA_Memory_Pin() {
        cudaHostUnregister(_ptr);
    }


private:
    void* _ptr;
};


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

    cudaError_t write_sub(T const* src, size_t offset, size_t count, cudaStream_t stream) {
        return cudaMemcpyAsync(&d_buf[offset], &src[offset], count * sizeof(T), cudaMemcpyHostToDevice, stream);
    }

    cudaError_t read_sub(T* dst_base, size_t offset, size_t count, cudaStream_t stream) {
        return cudaMemcpyAsync(&dst_base[offset], &d_buf[offset], count * sizeof(T), cudaMemcpyDeviceToHost, stream);
    }

    operator T*() {
        return d_buf;
    }

    T* d_buf;
    size_t N;
};

class CUDA_Event_Recycler {
public:
    size_t get(cudaEvent_t* out) {
        size_t ret;
        *out = 0;
        if(cursor < event_handles.size()) {
            *out = event_handles[cursor];
        } else {
            event_handles.push_back(0);
            auto& ev = event_handles.back();
            ASSERT_CUDA_SUCCEEDED(cudaEventCreateWithFlags(&ev, cudaEventDefault));
            *out = ev;
        }
        ret = cursor;
        cursor++;
        return ret;
    }

    void flip() {
        cursor = 0;
    }

    ~CUDA_Event_Recycler() {
        for(size_t i = 0; i < event_handles.size(); i++) {
            ASSERT_CUDA_SUCCEEDED(cudaEventDestroy(event_handles[i]));
        }
    }

private:
    std::vector<cudaEvent_t> event_handles;
    size_t cursor = 0;
};
