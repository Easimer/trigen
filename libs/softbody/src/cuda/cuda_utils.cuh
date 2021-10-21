// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: utility functions for the CUDA computation backend
//

#pragma once

#include <cassert>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "cuda_memtrack.h"

#define CUDA_SUCCEEDED(HR, APICALL) ((HR = APICALL) == cudaSuccess)

inline void cuda_assert(char const* expr, size_t line, char const* file, char const* function, cudaError_t rc) {
    if(rc == cudaSuccess) {
        return;
    }

    char const* msg = cudaGetErrorString(rc);

    printf("[!] CUDA API CALL FAILED: %s:%zu: %s: Assertion `%s` failed. Result code: %d (%s)\n", file, line, function, expr, rc, msg);
    std::abort();
}

inline void cuda_assert(char const* expr, size_t line, char const* file, char const* function, CUresult rc) {
    if(rc == CUDA_SUCCESS) {
        return;
    }

    char const *msg, *name;
    cuGetErrorName(rc, &name);
    cuGetErrorString(rc, &msg);

    printf("[!] CUDA API CALL FAILED: %s:%zu: %s: Assertion `%s` failed. Result code: %d (%s, '%s')\n", file, line, function, expr, rc, name, msg);
    std::abort();
}

#ifdef NDEBUG
#define ASSERT_CUDA_SUCCEEDED(APICALL) (APICALL)
#else
#define _ASSERT_CUDA_SUCCEEDED(expr) #expr
#define ASSERT_CUDA_SUCCEEDED(APICALL) cuda_assert((char const*)_ASSERT_CUDA_SUCCEEDED(APICALL), (size_t) __LINE__, (char const*) __FILE__, (char const*) __FUNCTION__, APICALL)
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
    CUDA_Memory_Pin() : _ptr(nullptr) {}

    CUDA_Memory_Pin(void const* ptr, size_t size) : _ptr(ptr) {
        ASSERT_CUDA_SUCCEEDED(cudaHostRegister((void*)ptr, size, cudaHostRegisterDefault));
    }

    template<typename T>
    CUDA_Memory_Pin(std::vector<T> const& v) : _ptr(v.data()) {
        // NOTE(danielm): you must be careful not to push items into the vector
        // since a vector grow might move the memory away from the current address.
        auto size = v.size() * sizeof(T);
        ASSERT_CUDA_SUCCEEDED(cudaHostRegister((void*)_ptr, size, cudaHostRegisterDefault));
    }

    CUDA_Memory_Pin& operator=(CUDA_Memory_Pin&& other) {
        if(_ptr != nullptr) {
            ASSERT_CUDA_SUCCEEDED(cudaHostUnregister((void*)_ptr));
            _ptr = nullptr;
        }

        std::swap(_ptr, other._ptr);

        return *this;
    }

    CUDA_Memory_Pin& operator=(CUDA_Memory_Pin const&) = delete;

    ~CUDA_Memory_Pin() {
        if(_ptr != nullptr) {
            ASSERT_CUDA_SUCCEEDED(cudaHostUnregister((void*)_ptr));
        }
    }


private:
    void const* _ptr;
};

template<typename T>
struct CUDA_Array_Base {
    CUDA_Array_Base() : d_buf(nullptr), N(0) {
    }

    CUDA_Array_Base(size_t N) : d_buf(nullptr), N(N) {
        if(N != 0) {
            ASSERT_CUDA_SUCCEEDED(cudaMalloc(&d_buf, N * sizeof(T)));
        }
    }

    CUDA_Array_Base(CUDA_Array_Base const& other) = delete;

    CUDA_Array_Base(CUDA_Array_Base&& other) : d_buf(nullptr), N(0) {
        std::swap(d_buf, other.d_buf);
        std::swap(N, other.N);
    }

    ~CUDA_Array_Base() {
        if(d_buf != nullptr) {
            ASSERT_CUDA_SUCCEEDED(cudaFree(d_buf));
        }
    }

    CUDA_Array_Base& operator=(CUDA_Array_Base const& other) = delete;

    CUDA_Array_Base& operator=(CUDA_Array_Base&& other) {
        if(!is_empty()) {
            ASSERT_CUDA_SUCCEEDED(cudaFree(d_buf));
        }

        d_buf = nullptr;
        N = 0;

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
        assert(!is_empty());
        return cudaMemcpyAsync(d_buf, src, bytes(), cudaMemcpyHostToDevice, stream);
    }

    cudaError_t read_async(T* dst, cudaStream_t stream) const {
        assert(!is_empty());
        return cudaMemcpyAsync(dst, d_buf, bytes(), cudaMemcpyDeviceToHost, stream);
    }

    cudaError_t write_sub(T const* src, size_t offset, size_t count, cudaStream_t stream) {
        assert(!is_empty());
        return cudaMemcpyAsync(&d_buf[offset], &src[offset], count * sizeof(T), cudaMemcpyHostToDevice, stream);
    }

    cudaError_t read_sub(T* dst_base, size_t offset, size_t count, cudaStream_t stream) const {
        assert(!is_empty());
        return cudaMemcpyAsync(&dst_base[offset], &d_buf[offset], count * sizeof(T), cudaMemcpyDeviceToHost, stream);
    }

    CUDA_Array_Base<T> duplicate(cudaStream_t stream) const {
        CUDA_Array_Base<T> ret(N);
        ASSERT_CUDA_SUCCEEDED(cudaMemcpyAsync(ret.d_buf, d_buf, N * sizeof(T), cudaMemcpyDeviceToDevice, stream));
        return ret;
    }

    operator T*() {
        return d_buf;
    }

    operator T const*() const {
        return d_buf;
    }

    T* d_buf;
    size_t N;
};

template<typename T, typename Tag = void>
struct CUDA_Array : public CUDA_Array_Base<T> {
    CUDA_Array() : CUDA_Array_Base<T>() {}

    CUDA_Array(size_t N) : CUDA_Array_Base<T>(N) {}

    CUDA_Array(CUDA_Array const& other) = delete;

    CUDA_Array(CUDA_Array&& other) : CUDA_Array_Base<T>(std::move(other)) {}

    CUDA_Array& operator=(CUDA_Array&& other) {
        CUDA_Array_Base<T>::operator=(std::move(other));
        return *this;
    }

    CUDA_Array<T>& untag() { return reinterpret_cast<CUDA_Array<T>&>(*this); }
    CUDA_Array<T> const& untag() const { return reinterpret_cast<CUDA_Array<T> const&>(*this); }

    T *operator->() {
        return d_buf;
    }
};

class CUDA_Event_Recycler {
public:
    size_t get(cudaEvent_t* out) {
        size_t ret;
        *out = 0;
        if(cursor < event_handles.size()) {
            *out = event_handles[cursor];
        } else {
            cudaEvent_t ev;
            ASSERT_CUDA_SUCCEEDED(cudaEventCreateWithFlags(&ev, cudaEventDefault));
            event_handles.push_back(ev);
            *out = ev;
        }
        ret = cursor;
        cursor++;
        return ret;
    }

    void flip() {
        cursor = 0;
    }

    CUDA_Event_Recycler()
        : event_handles(0) {
        event_handles.reserve(16);
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
