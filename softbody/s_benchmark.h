// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: benchmarking utils
//

#pragma once

#if SB_BENCHMARK

#include <chrono>
#define DECLARE_BENCHMARK_BLOCK() \
    static std::chrono::milliseconds sbbm_dur; \
    static size_t sbbm_count = 0;

#define BEGIN_BENCHMARK() \
    auto const sbbm_begin = std::chrono::high_resolution_clock::now();

#define END_BENCHMARK() \
    auto const sbbm_end = std::chrono::high_resolution_clock::now();

#define PRINT_BENCHMARK_RESULT_MASKED(mask) \
    auto const sbbm_diff = sbbm_end - sbbm_begin; \
    sbbm_dur = std::chrono::duration_cast<std::chrono::milliseconds>(sbbm_diff); \
    sbbm_count++; \
    if((sbbm_count & (mask)) == 0) { \
        printf("sb: benchmark: %s %llu ms\n", __func__, sbbm_dur.count()); \
    }

#define PRINT_BENCHMARK_RESULT() PRINT_BENCHMARK_RESULT_MASKED(0x00000000)

#endif /* SB_BENCHMARK */
