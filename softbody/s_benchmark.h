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
    static decltype(cpu_rdtsc()) rdtsc_sum = 0; \
    static size_t rdtsc_count = 0;

#define BEGIN_BENCHMARK() \
    auto const rdtsc_begin = cpu_rdtsc();

#define END_BENCHMARK() \
    auto const rdtsc_end = cpu_rdtsc();

#define PRINT_BENCHMARK_RESULT() \
    auto const rdtsc_diff = rdtsc_end - rdtsc_begin; \
    rdtsc_sum += rdtsc_diff; \
    rdtsc_count++; \
    printf("sb: benchmark: %s %f kilocycles\n", __func__, (double)rdtsc_sum / (double)rdtsc_count / 1000.0f);

#else
#define cpu_rdtsc() (0)
#endif /* SB_BENCHMARK */
