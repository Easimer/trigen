// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: computation backends
//

#pragma once

#include "softbody.h"
#include "s_simulation.h"

class Softbody_Simulation;

class ICompute_Backend {
public:
    virtual ~ICompute_Backend() {}

    virtual void do_one_iteration_of_shape_matching_constraint_resolution(System_State& sim, float phdt) = 0;
};

sb::Unique_Ptr<ICompute_Backend> Make_Reference_Backend();
sb::Unique_Ptr<ICompute_Backend> Make_CL_Backend();

inline sb::Unique_Ptr<ICompute_Backend> Make_Compute_Backend() {
    auto ret = Make_CL_Backend();

    if (ret == NULL) {
        ret = Make_Reference_Backend();
    }

    return ret;
}
