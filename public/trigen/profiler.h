// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: profiling utils
//

#pragma once

#ifdef PROFILER_ENABLE
#ifdef _NDEBUG
#define PROFILER_ENABLED (1)
#endif /* _NDEBUG */
#else
#define PROFILER_ENABLED (0)
#endif /* PROFILER_ENABLE */

#if PROFILER_ENABLED
#include <cstdio>
#include <SDL.h>

#define SCOPE_BENCHMARK() Scope_Benchmark __bm(__FUNCTION__)
#else
#define SCOPE_BENCHMARK()
#endif

struct Scope_Benchmark {
#if PROFILER_ENABLED
    Scope_Benchmark(char const* pszScopeName) : uiStart(SDL_GetPerformanceCounter()), pszScopeName(pszScopeName) {}

    ~Scope_Benchmark() {
        auto const uiEnd = SDL_GetPerformanceCounter();
        auto const flElapsed = (uiEnd - uiStart) / (double)SDL_GetPerformanceFrequency();
        printf("%s took %f milliseconds\n", pszScopeName, flElapsed * 1000);
    }

    char const* pszScopeName;
    Uint64 const uiStart;
#endif /* PROFILER_ENABLED */
};

#undef PROFILER_ENABLED
