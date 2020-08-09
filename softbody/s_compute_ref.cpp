// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: reference computation backend
//

#include "stdafx.h"
#include <cassert>
#include <array>
#include <algorithm>
#include "softbody.h"
#include "l_iterators.h"
#include "s_compute_backend.h"
#define SB_BENCHMARK (1)
#include "s_benchmark.h"
#include "m_utils.h"
#include <glm/gtx/matrix_operation.hpp>

#define NUMBER_OF_CLUSTERS(idx) (s.edges[(idx)].size() + 1)

class Compute_CPU_Single_Threaded : public ICompute_Backend {
    float mass_of_particle(System_State& s, unsigned i) const {
        auto const d_i = s.density[i];
        auto const s_i = s.size[i];
        auto const m_i = (4.f / 3.f) * glm::pi<float>() * s_i.x * s_i.y * s_i.z * d_i;
        return m_i;
    }

    size_t particle_count(System_State& s) const {
        return s.position.size();
    }

    void begin_new_frame(System_State const& sim) override {}

    void do_one_iteration_of_shape_matching_constraint_resolution(
        System_State& s,
        float phdt
    ) override {
        DECLARE_BENCHMARK_BLOCK();
        BEGIN_BENCHMARK();

        // shape matching constraint
        for (unsigned i = 0; i < particle_count(s); i++) {
            std::array<unsigned, 1> me{ i };
            auto& neighbors = s.edges[i];
            auto neighbors_and_me = iterator_union(neighbors.begin(), neighbors.end(), me.begin(), me.end());

            // Sum particle weights in the current cluster
            auto M = std::accumulate(
                neighbors.begin(), neighbors.end(),
                mass_of_particle(s, i),
                [&](float acc, unsigned idx) {
                    return acc + mass_of_particle(s, idx);
                }
            );

            assert(M != 0);

            auto invRest = s.bind_pose_inverse_bind_pose[i];
            auto com0 = s.bind_pose_center_of_mass[i];

            // Center of mass calculated using the predicted positions
            auto com_cur = std::accumulate(
                neighbors.begin(), neighbors.end(),
                mass_of_particle(s, i) * s.predicted_position[i],
                [&](Vec4 const& acc, unsigned idx) {
                    return acc + mass_of_particle(s, idx) * s.predicted_position[idx];
                }
            ) / M;

            s.center_of_mass[i] = com_cur;

            // Calculates the moment matrix of a single particle
            auto calc_A_i = [&](unsigned i) -> Mat4 {
                auto m_i = mass_of_particle(s, i);
                auto A_i = 1.0f / 5.0f * glm::diagonal4x4(s.size[i] * s.size[i]) * Mat4(s.orientation[i]);

                return m_i * (A_i + glm::outerProduct(s.predicted_position[i], s.bind_pose[i]) - glm::outerProduct(com_cur, com0));
            };

            // Calculate the cluster moment matrix
            auto A = std::accumulate(
                neighbors.begin(), neighbors.end(),
                calc_A_i(i),
                [&](Mat4 const& acc, unsigned idx) -> Mat4 {
                    return acc + calc_A_i(idx);
                }
            ) * invRest;

            // Extract the rotational part of A which is the least squares optimal
            // rotation that transforms the original bind-pose configuration into
            // the current configuration
            Quat R = s.predicted_orientation[i];
            mueller_rotation_extraction(A, R);

            float const stiffness = 1;

            // NOTE(danielm): not sure if we need to correct every particle in the
            // current cluster/group other than the group owner.
            // Works either way, but the commented out way would make parallelization
            // painful.
            // for (auto idx : neighbors_and_me) {
            {
                auto idx = i;
                // Bind pose position relative to the center of mass
                auto pos_bind = s.bind_pose[idx] - com0;
                // Current position relative to the center of mass
                auto d = s.predicted_position[idx] - com_cur;
                // Rotate the bind pose position relative to the CoM
                auto pos_bind_rot = R * pos_bind;
                // Our goal position
                auto goal = com_cur + pos_bind_rot;
                // Number of clusters this particle is a member of
                auto numClusters = NUMBER_OF_CLUSTERS(idx);
                auto correction = (goal - s.predicted_position[idx]) * stiffness;
                // The correction must be divided by the number of clusters this particle is a member of
                s.predicted_position[idx] += (1.0f / (float)numClusters) * correction;
                s.goal_position[idx] = goal;
            }

            s.predicted_orientation[i] = R;
        }

        END_BENCHMARK();
        PRINT_BENCHMARK_RESULT();
    }
};

sb::Unique_Ptr<ICompute_Backend> Make_Reference_Backend() {
    return std::make_unique<Compute_CPU_Single_Threaded>();
}
