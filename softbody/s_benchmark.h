// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: benchmarking utils
//

#pragma once

#if SB_BENCHMARK

#ifndef SB_BENCHMARK_UNITS
#define SB_BENCHMARK_UNITS milliseconds
#define SB_BENCHMARK_UNITS_STR "ms"
#endif

#include <chrono>
#define DECLARE_BENCHMARK_BLOCK() \
    static std::chrono::SB_BENCHMARK_UNITS sbbm_dur; \
    static size_t sbbm_count = 0;

#define BEGIN_BENCHMARK() \
    auto const sbbm_begin = std::chrono::high_resolution_clock::now();

#define END_BENCHMARK() \
    auto const sbbm_end = std::chrono::high_resolution_clock::now();

#define _XX(s) #s

#define PRINT_BENCHMARK_RESULT_MASKED(mask) \
    auto const sbbm_diff = sbbm_end - sbbm_begin; \
    sbbm_dur = std::chrono::duration_cast<std::chrono::SB_BENCHMARK_UNITS>(sbbm_diff); \
    sbbm_count++; \
    if((sbbm_count & (mask)) == 0) { \
        printf("sb: benchmark: file='%s' func='%s' time='%llu' units='" SB_BENCHMARK_UNITS_STR "'\n", __FILE__, __func__, sbbm_dur.count()); \
    }


#define PRINT_BENCHMARK_RESULT() PRINT_BENCHMARK_RESULT_MASKED(0x00000000)

#else /* SB_BENCHMARK */

#define DECLARE_BENCHMARK_BLOCK()
#define BEGIN_BENCHMARK()

#define END_BENCHMARK()

#define PRINT_BENCHMARK_RESULT_MASKED(mask)

#define PRINT_BENCHMARK_RESULT()

#endif /* SB_BENCHMARK */
