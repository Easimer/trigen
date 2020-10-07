// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: benchmark declaration
//

#include "stdafx.h"
#include <softbody.h>
#include "benchmark.h"

Benchmark Benchmark::make_benchmark(sb::Compute_Preference backend) {
    sb::Config cfg;
    cfg.ext = sb::Extension::Debug_Cloth;
    cfg.compute_preference = backend;

    auto sim = sb::create_simulation(cfg);

    return Benchmark(std::move(sim));
}

void Benchmark::run(float total_time) {
    double time_left = total_time;
    double const step = 1 / 30.0;
    while(time_left >= 0) {
        sim->step(step);
        time_left -= step;
    }
}
