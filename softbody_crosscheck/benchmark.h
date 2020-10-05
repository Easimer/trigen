// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: benchmark declaration
//

#pragma once

#include <softbody.h>

class Benchmark {
public:
    void run(float total_time);
    static Benchmark make_benchmark(sb::Compute_Preference backend);

private:
    Benchmark(sb::Unique_Ptr<sb::ISoftbody_Simulation>&& sim)
        : sim(std::move(sim)) {
    }
    
private:
    sb::Unique_Ptr<sb::ISoftbody_Simulation> sim;
};
