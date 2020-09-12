// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: computation backends
//

#pragma once

#include "common.h"

class Softbody_Simulation;
class ICompute_Backend_Complete;

class ICompute_Backend {
public:
    virtual ~ICompute_Backend() {}

    virtual void begin_new_frame(System_State const& sim) = 0;
    virtual void do_one_iteration_of_shape_matching_constraint_resolution(System_State& sim, float phdt) = 0;
};

sb::Unique_Ptr<ICompute_Backend> Make_Reference_Backend();
sb::Unique_Ptr<ICompute_Backend> Make_CL_Backend();
sb::Unique_Ptr<ICompute_Backend> Make_CUDA_Backend();

inline sb::Unique_Ptr<ICompute_Backend> Make_Compute_Backend() {
#if SOFTBODY_CUDA_ENABLED
    // Try creating a CUDA compute backend
    auto ret = Make_CUDA_Backend();

    if(ret == NULL) {
        // Try to fallback to OpenCL
        ret = Make_CL_Backend();
    }
#else
    auto ret = Make_CL_Backend();
#endif

    if (ret == NULL) {
        // Fallback to CPU backend
        ret = Make_Reference_Backend();
    }

    return ret;
}
