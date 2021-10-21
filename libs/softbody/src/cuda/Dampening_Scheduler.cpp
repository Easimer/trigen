// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: Dampening module
//

#include "../stdafx.h"
#include "Dampening_Scheduler.h"
#include <cstdio>

namespace Dampening {
void
Scheduler::compute_internal_forces(System_State &s) {
    auto N = s.position.size();
    Predicted_Position_Buffer pred_pos(N);
    Position_Buffer pos(N);
    Size_Buffer size(N);
    Density_Buffer density(N);
    Velocity_Buffer velocity(N);
    CUDA_Array<Centers_Of_Mass> centersOfMass(1);
    on_stream<Stream::CopyToDev>([&](cudaStream_t stream) {
        ASSERT_CUDA_SUCCEEDED(size.write_async(
            (float4 *)s.size.data(), stream));
        ASSERT_CUDA_SUCCEEDED(density.write_async(
            (float *)s.density.data(), stream));
    });
    insert_dependency<Stream::CopyToDev, Stream::Masses>(evr);
    on_stream<Stream::CopyToDev>([&](cudaStream_t stream) {
        ASSERT_CUDA_SUCCEEDED(pred_pos.write_async(
            (float4 *)s.predicted_position.data(), stream));
        ASSERT_CUDA_SUCCEEDED(pos.write_async(
            (float4 *)s.position.data(), stream));
    });
    insert_dependency<Stream::CopyToDev, Stream::Velocities>(evr);
    do_velocities(N, pos, pred_pos, velocity);
    insert_dependency<Stream::Velocities, Stream::DampeningForces>(evr);

    insert_dependency<Stream::CopyToDev, Stream::CentersOfMass>(evr);
    float M = do_masses(N, size, density);

    insert_dependency<Stream::Masses, Stream::CentersOfMass>(evr);
    do_centers_of_mass(N, M, pos, pred_pos, centersOfMass);

}
}
