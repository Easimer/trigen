// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: Dampening module
//

#pragma once

#include "CUDA_Scheduler.h"
#include "BufferTypes.h"

#include "../system_state.h"

namespace Dampening {
    using Velocity_Buffer = CUDA_Array<float4, struct Velocity_Buffer_Tag>;
    enum class Stream : size_t {
        CopyToDev,
        Masses,
        Velocities,
        CentersOfMass,
        DampeningForces,
        SumForces,
        InternalForces,

        Max
    };

    struct Centers_Of_Mass {
        float4 com0;
        float4 com1;
    };

    class Scheduler : public CUDA_Scheduler<Stream, Stream::Max> {
    public:
        Scheduler(std::array<cudaStream_t, (size_t)Stream::Max>&& streams)
            : CUDA_Scheduler(std::move(streams)) {
        }
        void
        compute_internal_forces(System_State &s);

        protected:
        float
        do_masses(
            int N,
            Size_Buffer const &size,
            Density_Buffer const &density);

        void
        do_velocities(
            int N,
            Position_Buffer const &pos,
            Predicted_Position_Buffer const &pred_pos,
            Velocity_Buffer &velocity);

        void
        do_centers_of_mass(
            int N,
            float M,
            Position_Buffer const &pos,
            Predicted_Position_Buffer const &pred_pos,
            CUDA_Array<Centers_Of_Mass> &centersOfMass
        );

    private:
        CUDA_Event_Recycler evr;
    };
    }


