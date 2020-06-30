// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: softbody simulation
//

#include "stdafx.h"
#include "softbody.h"
#include "m_utils.h"
#include "s_simulation.h"
#include <cstdlib>

#define PHYSICS_STEP (1.0f / 25.0f)
#define SIM_SIZE_LIMIT (128)

// #define DEBUG_TETRAHEDRON
#ifdef DEBUG_TETRAHEDRON
#define DISABLE_GROWTH
#define DISABLE_PHOTOTROPISM
#endif /* defined(DEBUG_TETRAHEDRON) */

// #define DISABLE_GROWTH
// #define DISABLE_PHOTOTROPISM

#define DEFER_LAMBDA(lambda)                \
{                                           \
    Lock_Guard g(deferred_lock);            \
    deferred.push_back(std::move(lambda));   \
}

static float randf() {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

void Softbody_Simulation::initialize(sb::Config const& configuration) {
    auto o = configuration.seed_position;
    auto siz = Vec3(1, 1, 2);
    auto idx_root = add_particle(o, siz, 1);
#ifdef DEBUG_TETRAHEDRON
    auto idx_t0 = add_particle(o + Vec3(-1, 2, 1), siz, 1);
    auto idx_t1 = add_particle(o + Vec3(+1, 2, 1), siz, 1);
    auto idx_t2 = add_particle(o + Vec3(0, 2, -1), siz, 1);

    connect_particles(idx_root, idx_t0);
    connect_particles(idx_root, idx_t1);
    connect_particles(idx_root, idx_t2);
    connect_particles(idx_t0, idx_t1);
    connect_particles(idx_t1, idx_t2);
    connect_particles(idx_t0, idx_t2);
#endif /* DEBUG_TETRAHEDRON */

    params = configuration;
}

void Softbody_Simulation::predict_positions(float dt) {
    auto const N = position.size();

    predicted_position.resize(N);

    auto const R = range(0, N);
    std::for_each(std::execution::par, R.begin(), R.end(), [&](unsigned i) {
        auto p_age = (age[i] += dt);
        auto stiffness = glm::min(1.0f, params.stiffness + p_age / params.aging_rate);

        auto a = Vec3();
        auto const a_g = Vec3(0, -1, 0);
        auto const goal_dir = (goal_position[i] - position[i]);
        auto const a_goal = goal_dir * dt;
        a += a_g;
        a += 64.0f * stiffness * a_goal;
        auto const v = velocity[i] + dt * a;
        velocity[i] = v;
        Vec3 const x_p = position[i] + dt * velocity[i] + (dt * dt / 2) * a;

        // TODO: set an upper bound for density
        // TODO: density rate of change parameter
        density[i] = glm::min(8.0f, density[i] += dt * params.aging_rate);

        predicted_position[i] = x_p;

#ifndef DISABLE_PHOTOTROPISM
        // Phototropism
        auto v_forward = glm::normalize(orientation[i] * Vec3(0, 1, 0) * glm::inverse(orientation[i]));
        auto v_light = glm::normalize(light_source - position[i]);
        auto v_light_axis = glm::cross(v_light, v_forward);
        auto O = 0.0f; // TODO: detect occlusion, probably by shadow mapping
        auto angle_light = (1 - O) * params.phototropism_response_strength * dt;
        auto Q_light = glm::normalize(Quat(angle_light, v_light_axis));
        angular_velocity[i] = Q_light * angular_velocity[i] * glm::inverse(Q_light);
        orientation[i] = Q_light * orientation[i];
#endif /* !defined(DISABLE_PHOTOTROPISM) */

        auto ang_vel = angular_velocity[i];
        auto orient = orientation[i];
        auto len_ang_vel = glm::length(ang_vel);
        Quat orient_temp = orient;
        if (len_ang_vel >= glm::epsilon<float>()) {
            auto comp = (len_ang_vel * dt) / 2;
            orient_temp = Quat(glm::cos(comp), (ang_vel / len_ang_vel) * glm::sin(comp));
        }

        auto delta_orient_angle = glm::angle(orient_temp * glm::inverse(orient));
        auto delta_orient_axis = glm::axis(orient_temp * glm::inverse(orient));

        if (glm::abs(delta_orient_angle) >= 0.05f) {
            angular_velocity[i] = (delta_orient_angle / dt) * delta_orient_axis;
        }

        orientation[i] = orient_temp;
    });
}

void Softbody_Simulation::simulate_group(unsigned pidx, float dt) {
    auto const& neighbors = edges[pidx];
    auto owner_pos = predicted_position[pidx];
    auto& owner_rest_pos = rest_position[pidx];

    auto const [owner_mass, masses] = [&]() {
        Vector<float> masses;
        for (auto i : neighbors) {
            masses.push_back(mass_of_particle(i));
        }
        auto owner_mass = mass_of_particle(pidx);
        return std::make_tuple(owner_mass, masses);
    }();

    auto const W = [owner_mass = owner_mass, masses = masses]() {
        float sum = 0;
        for (auto m : masses) {
            sum += m;
        }
        sum += owner_mass;
        return sum;
    }();

    auto const [c, c_rest] = [&]() {
        auto c = Vec3();
        auto c_rest = Vec3();

        for (auto i : neighbors) {
            auto const m_i = mass_of_particle(i);
            c += (m_i / W) * predicted_position[i];
            c_rest += (m_i / W) * rest_position[i];
        }
        auto const m_c = mass_of_particle(pidx);
        c += (m_c / W) * owner_pos;
        c_rest += (m_c / W) * owner_rest_pos;

        return std::make_tuple(c, c_rest);
    }();

    auto group = Particle_Group{ pidx, owner_mass, neighbors, masses, W, c, c_rest, Mat3() };
    calculate_orientation_matrix(&group);
    auto const x_t = group.orient * (owner_rest_pos - group.c_rest) + group.c;
    auto const& owner_predicted = predicted_position[pidx];

    position[pidx] = owner_predicted;

    center_of_mass[pidx] = c;
    rest_center_of_mass[pidx] = c_rest;

    goal_position[pidx] = x_t;

#ifndef DISABLE_GROWTH
    // TODO(danielm): novekedesi rata
    // Novesztjuk az agat
    auto g = 1.0f; // 1/sec
    auto prob_branching = 0.25f;
    auto& r = size[pidx].x;
    if (r < 1.5f) {
        size[pidx] += Vec3(g * dt, g * dt, 0);

        // Ha tulleptuk a meret-limitet, novesszunk uj agat
        if (r >= 1.5f && position.size() < SIM_SIZE_LIMIT) {
            auto lateral_chance = randf();
            constexpr auto new_size = Vec3(0.5f, 0.5f, 2.0f);
            auto longest_axis = longest_axis_normalized(size[pidx]);
            auto new_longest_axis = longest_axis_normalized(new_size);
            if (lateral_chance < params.branching_probability) {
                // Oldalagat novesszuk
                auto angle = (2 * randf() - 1) * params.branch_angle_variance;
                auto x = 2 * randf() - 1;
                auto y = 2 * randf() - 1;
                auto z = 2 * randf() - 1;
                auto axis = glm::normalize(Vec3(x, y, z));
                auto bud_rot_offset = glm::angleAxis(angle, axis);
                auto lateral_orientation = bud_rot_offset * orientation[pidx];
                auto l_pos = position[pidx]
                    + orientation[pidx] * (longest_axis / 2.0f) * glm::inverse(orientation[pidx])
                    + lateral_orientation * (new_longest_axis / 2.0f) * glm::inverse(lateral_orientation);

                auto func_add = [&, l_pos, new_size, pidx, lateral_orientation]() {
                    auto l_idx = add_particle(l_pos, new_size, 1.0f);
                    lateral_bud[pidx] = l_idx;
                    connect_particles(pidx, l_idx);
                    orientation[l_idx] = lateral_orientation;
                };

                DEFER_LAMBDA(func_add);
            }

            // Csucsot novesszuk
            auto pos = position[pidx]
                + orientation[pidx] * (longest_axis / 2.0f) * glm::inverse(orientation[pidx])
                + orientation[pidx] * (new_longest_axis / 2.0f) * glm::inverse(orientation[pidx]);
            auto func_add = [&, pos, new_size, pidx]() {
                auto a_idx = add_particle(pos, new_size, 1.0f);
                apical_child[pidx] = a_idx;
                connect_particles(pidx, a_idx);
            };

            DEFER_LAMBDA(func_add);
        }
    }
#endif /* !defined(DISABLE_GROWTH) */
}

unsigned Softbody_Simulation::add_particle(Vec3 const& p_pos, Vec3 const& p_size, float p_density) {
    assert(!assert_parallel);
    assert(p_density >= 0.0f && p_density <= 1.0f);
    Vec3 zero(0, 0, 0);
    unsigned const index = position.size();
    position.push_back(p_pos);
    rest_position.push_back(p_pos);
    velocity.push_back(zero);
    angular_velocity.push_back(zero);
    goal_position.push_back(p_pos);
    center_of_mass.push_back(zero);
    rest_center_of_mass.push_back(zero);
    size.push_back(p_size);
    density.push_back(p_density);
    orientation.push_back(Quat(1.0f, 0.0f, 0.0f, 0.0f));
    age.push_back(0);
    edges[index] = {};

    return index;
}

void Softbody_Simulation::connect_particles(unsigned a, unsigned b) {
    assert(!assert_parallel);
    assert(a < position.size());
    assert(b < position.size());

    edges[a].push_back(b);
    edges[b].push_back(a);
}

float Softbody_Simulation::mass_of_particle(unsigned i) {
    auto const d_i = density[i];
    auto const s_i = size[i];
    auto const m_i = (4.f / 3.f) * glm::pi<float>() * s_i.x * s_i.y * s_i.z * d_i;
    return m_i;
}

void Softbody_Simulation::calculate_orientation_matrix(Particle_Group* group) {
    auto calculate_particle_matrix = [this](unsigned pidx) -> Mat3 {
        auto mass_i = this->mass_of_particle(pidx);
        auto const size = this->size[pidx];
        auto const a = size[0];
        auto const b = size[1];
        auto const c = size[2];
        auto const A_i = Mat3(a * a, 0, 0, 0, b * b, 0, 0, 0, c * c);
        return mass_i * (1 / 5.0f) * A_i;
    };

    auto calculate_group_matrix = [this, group, calculate_particle_matrix]() -> Mat3 {
        // sum (particle matrix + particle mass * (particle pos * particle rest pos)) - total group mass * (center * rest center)

        float const sum_masses = group->owner_mass + sum<float>(group->masses.begin(), group->masses.end());

        auto moment_sum =
            calculate_particle_matrix(group->owner)
            + this->mass_of_particle(group->owner) * glm::outerProduct(predicted_position[group->owner], rest_position[group->owner]);

        for (auto neighbor : group->neighbors) {
            moment_sum +=
                calculate_particle_matrix(neighbor)
                + this->mass_of_particle(neighbor) * glm::outerProduct(predicted_position[neighbor], rest_position[neighbor]);
        }

        return moment_sum - sum_masses * glm::outerProduct(group->c, group->c_rest);
    };

    auto is_null_matrix = [](Mat3 const& m) -> bool {
        float sum = 0;
        for (int i = 0; i < 3; i++) {
            sum += m[i].length();
        }

        return sum < glm::epsilon<float>();
    };

    auto const& owner_size = size[group->owner];

    auto const R = polar_decompose_r(calculate_group_matrix());

    group->orient = R;
}


Softbody_Simulation* sb::create_simulation(Config const& configuration) {
    auto ret = new Softbody_Simulation;

    ret->initialize(configuration);

    return ret;
}

void sb::destroy_simulation(Softbody_Simulation* s) {
    assert(s != NULL);

    if (s != NULL) {
        delete s;
    }
}

void sb::set_light_source_position(Softbody_Simulation* s, Vec3 const& pos) {
    assert(s != NULL);
    if (s != NULL) {
        s->light_source = pos;
    }
}

void sb::step(Softbody_Simulation* s, float delta_time) {
    assert(s != NULL);
    if (s != NULL) {
        s->time_accumulator += delta_time;

        // Nem per-frame szimulalunk, hanem fix idokozonkent, hogy ne valjon
        // instabilla a szimulacio
        while (s->time_accumulator > PHYSICS_STEP) {
            auto phdt = PHYSICS_STEP;
            auto p0 = s->position[0];

            s->predict_positions(phdt);

            if (s->time_accumulator > 8 * PHYSICS_STEP) {
                fprintf(stderr, "EXTREME LAG, ACC = %f\n", s->time_accumulator);
            }

            auto R = range(0, s->predicted_position.size());
            s->assert_parallel = true;
            std::for_each(std::execution::par, R.begin(), R.end(), [&](unsigned i) {
                s->simulate_group(i, phdt);
            });
            s->assert_parallel = false;

            // The creation of new particles is deferred until after all the
            // groups have been simulated
            for (auto& def_func : s->deferred) {
                def_func();
            }
            s->deferred.clear();

            s->position[0] = p0;

            s->time_accumulator -= PHYSICS_STEP;
        }
    }
}
