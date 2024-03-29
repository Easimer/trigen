// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: computation backends
//

#pragma once

#include "system_state.h"
#include "logger.h"

class Softbody_Simulation;
class ICompute_Backend_Complete;

class ICompute_Backend {
public:
    virtual ~ICompute_Backend() = default;

    virtual void set_debug_visualizer(sb::IDebug_Visualizer *pVisualizer) {}

    virtual void begin_new_frame(System_State const& sim) = 0;
    virtual void end_frame(System_State const& sim) {}

    virtual void predict(System_State& sim, float dt) = 0;
    virtual void integrate(System_State& sim, float dt) = 0;
    virtual void dampen(System_State& sim, float dt) = 0;

    virtual void generate_collision_constraints(System_State& sim) = 0;

    virtual void do_one_iteration_of_collision_constraint_resolution(System_State& sim, float phdt) = 0;
    virtual void do_one_iteration_of_fixed_constraint_resolution(System_State& sim, float phdt) = 0;
    virtual void do_one_iteration_of_distance_constraint_resolution(System_State& sim, float phdt) = 0;
    virtual void do_one_iteration_of_shape_matching_constraint_resolution(System_State& sim, float phdt) = 0;

    virtual void on_collider_added(System_State const& sim, sb::ISoftbody_Simulation::Collider_Handle handle) {}
    virtual void on_collider_removed(System_State const& sim, sb::ISoftbody_Simulation::Collider_Handle handle) {}
    virtual void on_collider_changed(System_State const& sim, sb::ISoftbody_Simulation::Collider_Handle handle) {}

    /**
     * Checks whether a line segment with two endpoints `from[i]` and `to[i]`
     * intersects the world geometry and puts the result (1 if it does and 0
     * otherwise) in `result[i]`.
     *
     * These three arrays must have equal lengths.
     */
    virtual void
    check_intersections(
        System_State const &sim,
        Vector<unsigned> &result,
        Vector<Vec3> const &from,
        Vector<Vec3> const &to)
        = 0;
};

sb::Unique_Ptr<ICompute_Backend> Make_Reference_Backend(ILogger* logger);
sb::Unique_Ptr<ICompute_Backend> Make_CL_Backend(ILogger* logger);
sb::Unique_Ptr<ICompute_Backend> Make_CUDA_Backend(ILogger* logger);

inline sb::Unique_Ptr<ICompute_Backend> Make_Compute_Backend(sb::Compute_Preference pref, ILogger* logger) {
    sb::Unique_Ptr<ICompute_Backend> ret;
    bool np = pref == sb::Compute_Preference::None;

    if (pref == sb::Compute_Preference::Reference) {
        return Make_Reference_Backend(logger);
    }

#if SOFTBODY_CUDA_ENABLED
    if (pref == sb::Compute_Preference::GPU_Proprietary || np) {
        ret = Make_CUDA_Backend(logger);
        if (ret != nullptr) {
            return ret;
        }
    }
#endif /* SOFTBODY_CUDA_ENABLED */

    if (
        (ret == nullptr && pref == sb::Compute_Preference::GPU_Proprietary) ||
        pref == sb::Compute_Preference::GPU_OpenCL ||
        np) {
        ret = Make_CL_Backend(logger);
        if (ret != nullptr) {
            return ret;
        }
    }

    return Make_Reference_Backend(logger);
}
