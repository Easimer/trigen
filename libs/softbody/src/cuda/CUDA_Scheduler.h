// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: CUDA work scheduler
//

#pragma once

#include <cstdio>
#include <array>
#include <functional>
#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_utils.cuh"

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
