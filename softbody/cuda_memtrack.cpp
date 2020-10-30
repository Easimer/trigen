// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: CUDA memtrack
//

#include <signal.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <list>
#include <vector>
#include <string>

// If you uncomment this memtrack will print all CUDA mallocs to the console
// #define MEMTRACK_VERBOSE

#ifdef SOFTBODY_CUDA_MEMTRACK
struct Frame {
    void *ptr;
    std::string sym;
};

struct Allocation {
    void *start, *end;
    bool page_locked;

    std::vector<Frame> backtrace;
};

#if __linux__
#include <execinfo.h>
static std::vector<Frame> generate_backtrace() { 
    std::vector<Frame> ret;

    void* ptrs[64];
    int n = backtrace(ptrs, 64);
    auto syms = backtrace_symbols(ptrs, n);

    for(int i = 0; i < n; i++) {
        ret.push_back({ (void*)ptrs[i], std::string(syms[i]) });
    }

    free(syms);

    return ret;
}
#else
static std::vector<Frame> generate_backtrace() { return {}; }
#endif

static std::list<Allocation> g_allocations;

static void handler(int sig, siginfo_t *si, void *) {
    void* addr = si->si_addr;

    printf("[ memtrack ] invalid memory access, addr=%p, checking allocations\n", addr);

    for(auto it = g_allocations.begin(); it != g_allocations.end(); ++it) {
        if(it->start <= addr && addr < it->end) {
            auto siz = (unsigned)((char*)it->end - (char*)it->start);
            printf("[ memtrack ] address is from allocation [%p -> %p, siz=%u, lock=%d]\n", it->start, it->end, siz, it->page_locked ? 1 : 0);
            printf("    allocated here:\n");

            for(int i = 0; i < it->backtrace.size(); i++) {
                auto& frame = it->backtrace[i];
                printf("#%d %p '%s'\n", i, frame.ptr, frame.sym.c_str());
            }
        }
    }

    printf("[ memtrack ] aborting...\n");
    std::abort();
}

static void attach_sighandler() {
    struct sigaction sa;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    sa.sa_sigaction = handler;
    if (sigaction(SIGSEGV, &sa, NULL) == -1) {
        printf("[ memtrack ] sigaction failed\n");
    }
}

namespace sb::CUDA::memtrack {
    void activate() {
        attach_sighandler();
    }

    void diagnose_address(void *p) {
        Allocation *src = NULL;
        for(auto& alloc : g_allocations) {
            if(alloc.start <= p && p < alloc.end) {
                src = &alloc;
                break;
            }
        }

        if(src == NULL) {
            printf("[ memtrack ] diagnostic for address %p: wasn't allocated using CUDA\n", p);
            return;
        }

        printf("[ memtrack ] diagnostic for address %p: %p-%p locked(%d); trace:\n", p, src->start, src->end, src->page_locked ? 1 : 0);
        for(int i = 0; i < src->backtrace.size(); i++) {
            auto& frame = src->backtrace[i];
            printf("#%d %p '%s'\n", i, frame.ptr, frame.sym.c_str());
        }
        printf("[ memtrack ] end of diagnostic\n");
    }

    static void add_allocation(void *start, void *end, bool page_locked, std::vector<Frame>&& trace) {
        g_allocations.push_front({ start, end, page_locked, std::move(trace) });

#if defined(MEMTRACK_VERBOSE)
        printf("[ memtrack ] alloc: start %p end %p page_locked %s\n", start, end, page_locked ? "true" : "false");
#endif
    }

    static void remove_allocation(void *start) {
        for(auto it = g_allocations.begin(); it != g_allocations.end(); ++it) {
            if(it->start == start) {
#if defined(MEMTRACK_VERBOSE)
                printf("[ memtrack ] alloc: start %p end %p page_locked %s\n", it->start, it->end, it->page_locked ? "true" : "false");
#endif
                g_allocations.erase(it);
                return;
            }
        }

        printf("[ memtrack ] free: start %p ENOENT\n", start);
    }

    CUresult _cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
        auto trace = generate_backtrace();
        auto ret = ::cuMemAlloc(dptr, bytesize);

        if(ret == CUDA_SUCCESS) {
            auto start = (char*)*dptr;
            add_allocation(start, start + bytesize, false, std::move(trace));
        }

        return ret;
    }

    CUresult _cuMemFree(CUdeviceptr dptr) {
        auto trace = generate_backtrace();
        auto ret = ::cuMemFree(dptr);

        if(ret == CUDA_SUCCESS) {
            remove_allocation((void*)dptr);
        }

        return ret;
    }

    CUresult _cuMemAllocHost(void **pp, size_t bytesize) {
        auto trace = generate_backtrace();
        auto ret = ::cuMemAllocHost(pp, bytesize);

        if(ret == CUDA_SUCCESS) {
            auto start = (char*)*pp;
            add_allocation(start, start + bytesize, true, std::move(trace));
        }

        return ret;
    }

    CUresult _cuMemFreeHost(void *p) {
        auto trace = generate_backtrace();
        auto ret = ::cuMemFreeHost(p);

        if(ret == CUDA_SUCCESS) {
            remove_allocation((void*)p);
        }

        return ret;
    }

    cudaError_t _cudaMalloc(void **devPtr, size_t size) {
        auto trace = generate_backtrace();
        auto ret = ::cudaMalloc(devPtr, size);

        if(ret == cudaSuccess) {
            auto start = (char*)*devPtr;
            add_allocation(start, start + size, false, std::move(trace));
        }

        return ret;
    }

    cudaError_t _cudaFree(void *devPtr) {
        auto trace = generate_backtrace();
        auto ret = ::cudaFree(devPtr);

        if(ret == cudaSuccess) {
            remove_allocation(devPtr);
        }

        return ret;
    }

    cudaError_t _cudaMallocHost(void **ptr, size_t size) {
        auto trace = generate_backtrace();
        auto ret = ::cudaMallocHost(ptr, size);

        if(ret == cudaSuccess) {
            auto start = (char*)*ptr;
            add_allocation(start, start + size, true, std::move(trace));
        }
        return ret;
    }

    cudaError_t _cudaFreeHost(void *ptr) {
        auto trace = generate_backtrace();
        auto ret = ::cudaFreeHost(ptr);

        if(ret == cudaSuccess) {
            remove_allocation(ptr);
        }

        return ret;
    }
}
#else
namespace sb::CUDA::memtrack {
    void activate() {}
    void diagnose_address(void *) {}
}
#endif
