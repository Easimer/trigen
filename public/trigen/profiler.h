// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: profiling utils
//

#pragma once
#include <cstdio>
#include <SDL.h>

#ifdef _NDEBUG
#define SCOPE_BENCHMARK() Scope_Benchmark __bm(__FUNCTION__)
#else
#define SCOPE_BENCHMARK()
#endif

struct Scope_Benchmark {
#ifdef _NDEBUG
    Scope_Benchmark(char const* pszScopeName) : uiStart(SDL_GetPerformanceCounter()), pszScopeName(pszScopeName) {}

    ~Scope_Benchmark() {
        auto const uiEnd = SDL_GetPerformanceCounter();
        auto const flElapsed = (uiEnd - uiStart) / (double)SDL_GetPerformanceFrequency();
        printf("%s took %f milliseconds\n", pszScopeName, flElapsed * 1000);
    }

    char const* pszScopeName;
    Uint64 const uiStart;
#endif
};
