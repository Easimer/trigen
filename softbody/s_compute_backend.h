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

inline sb::Unique_Ptr<ICompute_Backend> Make_Compute_Backend(sb::Compute_Preference pref) {
    bool np = pref == sb::Compute_Preference::None;

    if (pref == sb::Compute_Preference::Reference) {
        return Make_Reference_Backend();
    }

#if SOFTBODY_CUDA_ENABLED
    if (pref == sb::Compute_Preference::GPU_Proprietary || np) {
        auto ret = Make_CUDA_Backend();
        if (ret != nullptr) {
            return ret;
        }
    }
#endif /* SOFTBODY_CUDA_ENABLED */

    if (pref == sb::Compute_Preference::GPU_OpenCL || np) {
        auto ret = Make_CL_Backend();
        if (ret != nullptr) {
            return ret;
        }
    }

    return Make_Reference_Backend();
}
