// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: plant simulation
//

#include "stdafx.h"
#include "common.h"
#include "softbody.h"
#include "s_ext.h"
#include "m_utils.h"
#include "l_random.h"

class Plant_Simulation : public ISimulation_Extension {
public:
    Plant_Simulation(sb::Config const& params) : params(params) {
    }
private:
    sb::Config params;
    Rand_Float rnd;
    Map<unsigned, unsigned> parents;
    Map<unsigned, unsigned> apical_child;
    Map<unsigned, unsigned> lateral_bud;
    Map<index_t, Vec4> anchor_points;
    Dequeue<index_t> growing;

    void init(IParticle_Manager_Deferred* pman, System_State& s, float dt) override {
        // Create root
        pman->defer([&](IParticle_Manager* pman, System_State&) {
            constexpr auto new_size = Vec3(0.5f, 0.5f, 2.0f);
            auto o = params.seed_position;
            auto root0 = pman->add_init_particle(o - Vec3(0, 0.5, 0), new_size, 1);
            auto root1 = pman->add_init_particle(o, new_size, 1);
            pman->connect_particles(root0, root1);

            unsigned indices[] = { root0, root1 };

            pman->add_fixed_constraint(2, indices);
        });
    }


    void post_prediction(IParticle_Manager_Deferred* pman, System_State& s, float dt) override {
        auto const surface_adaption_min_dist = 8.0f;
        auto const attachment_min_dist = 0.05f;

        // For all particles
        for (index_t i = 0; i < s.position.size(); i++) {
            auto const p = s.predicted_position[i];
            if (anchor_points.count(i)) {
                auto ap = anchor_points[i];
                auto dist = distance(p, ap);
                if (dist <= attachment_min_dist) {
                    s.predicted_position[i] += dt * params.attachment_strength * (ap - s.predicted_position[i]);
                } else {
                    anchor_points.erase(i);
                }
            } else {
                System_State::SDF_Slot const* surface = NULL;
                float surface_dist = INFINITY;

                // Find the closest surface
                for (auto& C : s.colliders_sdf) {
                    auto dist = C.fun(p);
                    if (dist < surface_dist) {
                        surface = &C;
                        surface_dist = dist;
                    }
                }

                // If the surface is close enough, move towards it
                if (surface_dist < surface_adaption_min_dist) {
                    auto normal = sdf::normal(surface->fun, p);
                    auto surface = p - surface_dist * normal;

                    s.predicted_position[i] += dt * params.surface_adaption_strength * (surface - s.predicted_position[i]);

                    if (surface_dist < attachment_min_dist) {
                        anchor_points[i] = surface;
                    }
                }
            }
        }
    }

    void post_integration(IParticle_Manager_Deferred* pman_defer, System_State& s, float dt) override {
        growth(pman_defer, s, dt);
    }

    bool try_grow_branch(index_t pidx, IParticle_Manager_Deferred* pman_defer, System_State& s, float dt) {
        if (length(s.velocity[pidx]) > 2) return false;

        auto lateral_chance = rnd.normal();
        constexpr auto new_size = Vec3(0.25f, 0.25f, 2.0f);
        auto longest_axis = longest_axis_normalized(s.size[pidx]);
        auto new_longest_axis = longest_axis_normalized(new_size);

        auto parent = parents[pidx];
        auto branch_dir = normalize(s.bind_pose[pidx] - s.bind_pose[parent]);
        auto branch_len = 2.0f;

        if (lateral_chance < params.branching_probability) {
            auto angle = rnd.central() * params.branch_angle_variance;
            auto x = rnd.central();
            auto y = rnd.central();
            auto z = rnd.central();
            auto axis = glm::normalize(Vec3(x, y, z));
            auto bud_rot_offset = glm::angleAxis(angle, axis);
            auto lateral_branch_dir = bud_rot_offset * branch_dir * glm::conjugate(bud_rot_offset);

            auto pos = s.position[pidx] + branch_len * lateral_branch_dir;

            pman_defer->defer([&, pos, new_size, pidx](IParticle_Manager* pman, System_State& s) {
                auto l_idx = pman->add_particle(pos, new_size, 1.0f, pidx);
                lateral_bud[pidx] = l_idx;
                parents[l_idx] = pidx;
            });
        }
        auto pos = s.position[pidx] + branch_len * branch_dir;

        pman_defer->defer([&, pos, new_size, pidx](IParticle_Manager* pman, System_State& s) {
            auto a_idx = pman->add_particle(pos, new_size, 1.0f, pidx);
            apical_child[pidx] = a_idx;
            parents[a_idx] = pidx;
        });

        return true;
    }

    void growth(IParticle_Manager_Deferred* pman_defer, System_State& s, float dt) {
        auto g = 1.0f; // 1/sec
        auto prob_branching = 0.25f;
        auto const N = s.position.size();
        auto const max_size = 1.5f;

        if (N >= params.particle_count_limit) return;

        for (index_t pidx = 1; pidx < N; pidx++) {
            auto r = s.size[pidx].x;

            s.size[pidx] += Vec4(g * dt, g * dt, 0, 0);
            if (r < max_size) {
                r = s.size[pidx].x;

                // Amint tullepjuk a reszecske meret limitet, novesszunk uj agat
                if (r >= max_size && N < params.particle_count_limit) {
                    growing.push_back(pidx);
                }
            }
        }

        auto remain = growing.size();
        while (remain--) {
            auto pidx = growing.front();
            growing.pop_front();
            if (!try_grow_branch(pidx, pman_defer, s, dt)) {
                growing.push_back(pidx);
            }
        }
    }
};

sb::Unique_Ptr<ISimulation_Extension> Create_Extension_Plant_Simulation(sb::Extension kind, sb::Config const& params) {
    return std::make_unique<Plant_Simulation>(params);
}
