// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: plant simulation
//

#include "stdafx.h"
#include "common.h"
#include "softbody.h"
#include "s_ext.h"
#include "m_sdf.h"
#include "m_utils.h"
#include "l_random.h"

class Plant_Simulation : public ISimulation_Extension {
public:
    Plant_Simulation(sb::Config const& params) : params(params) {
    }
private:
    sb::Config params;
    Rand_Float rnd;

    void post_prediction(IParticle_Manager_Deferred* pman, System_State& s, float dt) override {
        return;
        auto const anchor_point_min_dist = 2.0f;
        auto const attachment_strength = 0.25f;
        for (index_t i = 0; i < s.position.size(); i++) {
            Vector<Vec3> anchor_points;

            auto p = s.predicted_position[i];
            for (auto& C : s.colliders_sdf) {
                auto t = C.fun(p);
                if (t < anchor_point_min_dist) {
                    auto normal = sdf::normal(C.fun, p);
                    auto anchor_point = p - t * normal;

                    s.predicted_position[i] += attachment_strength * (anchor_point - s.predicted_position[i]);
                }
            }
        }
    }

    void post_integration(IParticle_Manager_Deferred* pman_defer, System_State& s, float dt) override {
        growth(pman_defer, s, dt);
    }

    void growth(IParticle_Manager_Deferred* pman_defer, System_State& s, float dt) {
        auto g = 1.0f; // 1/sec
        auto prob_branching = 0.25f;
        auto const N = s.position.size();
        auto const max_size = 1.5f;

        if (N >= params.particle_count_limit) return;

        for (index_t pidx = 0; pidx < N; pidx++) {
            auto& r = s.size[pidx].x;

            if (r < max_size) {
                s.size[pidx] += Vec3(g * dt, g * dt, 0);

                // Amint tullepjuk a reszecske meret limitet, novesszunk uj agat
                if (r >= max_size && N < params.particle_count_limit) {
                    auto lateral_chance = rnd.normal();
                    constexpr auto new_size = Vec3(0.5f, 0.5f, 2.0f);
                    auto longest_axis = longest_axis_normalized(s.size[pidx]);
                    auto new_longest_axis = longest_axis_normalized(new_size);
                    if (lateral_chance < params.branching_probability) {
                        // Oldalagat novesszuk
                        auto angle = rnd.central() * params.branch_angle_variance;
                        auto x = rnd.central();
                        auto y = rnd.central();
                        auto z = rnd.central();
                        auto axis = glm::normalize(Vec3(x, y, z));
                        auto bud_rot_offset = glm::angleAxis(angle, axis);
                        auto lateral_orientation = bud_rot_offset * s.orientation[pidx];
                        auto l_pos = s.position[pidx]
                            + s.orientation[pidx] * (longest_axis / 2.0f) * glm::inverse(s.orientation[pidx])
                            + lateral_orientation * (new_longest_axis / 2.0f) * glm::inverse(lateral_orientation);

                        pman_defer->defer([&, l_pos, new_size, pidx, lateral_orientation](IParticle_Manager* pman, System_State& s) {
                            auto l_idx = pman->add_particle(l_pos, new_size, 1.0f, pidx);
                            s.lateral_bud[pidx] = l_idx;
                            s.orientation[l_idx] = lateral_orientation;
                        });
                    }

                    auto pos = s.position[pidx]
                        + s.orientation[pidx] * Vec3(1, 0, 0) * glm::inverse(s.orientation[pidx]);
                        // + s.orientation[pidx] * (new_longest_axis / 2.0f) * glm::inverse(s.orientation[pidx]);

                    pman_defer->defer([&, new_size, pidx](IParticle_Manager* pman, System_State& s) {
                        auto pos = s.position[pidx] + Vec3(0, 0.5, 0);

                        auto a_idx = pman->add_particle(pos, new_size, 1.0f, pidx);
                        s.apical_child[pidx] = a_idx;
                    });
                }
            }
        }
    }
};

sb::Unique_Ptr<ISimulation_Extension> Create_Extension_Plant_Simulation(sb::Extension kind, sb::Config const& params) {
    return std::make_unique<Plant_Simulation>(params);
}
