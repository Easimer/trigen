// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: benchmarking utils
//

#pragma once

#if SB_BENCHMARK


#if _WIN32
#define cpu_rdtsc() (__rdtsc())
#else
#define cpu_rdtsc() (0)
#endif /* PLATFORM */

#define DECLARE_BENCHMARK_BLOCK() \
    static decltype(cpu_rdtsc()) rdtsc = 0; \
    static size_t rdtsc_count = 0;

#define BEGIN_BENCHMARK() \
    auto const rdtsc_begin = cpu_rdtsc();

#define END_BENCHMARK() \
    auto const rdtsc_end = cpu_rdtsc();

#define PRINT_BENCHMARK_RESULT_MASKED(mask) \
    auto const rdtsc_diff = rdtsc_end - rdtsc_begin; \
    rdtsc = rdtsc_diff; \
    rdtsc_count++; \
    if((rdtsc_count & (mask)) == 0) { \
        printf("sb: benchmark: %s %f kilocycles\n", __func__, (double)rdtsc / 1000.0f); \
    }

#define PRINT_BENCHMARK_RESULT() PRINT_BENCHMARK_RESULT_MASKED(0x00000000)

#else
#define cpu_rdtsc() (0)
#endif /* SB_BENCHMARK */
