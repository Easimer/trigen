// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: softbody simulation
//

#include "stdafx.h"
#include "softbody.h"
#include "m_utils.h"
#include "s_simulation.h"
#include <cstdlib>
#include <array>

#define PHYSICS_STEP (1.0f / 25.0f)
#define TAU (PHYSICS_STEP)
#define SIM_SIZE_LIMIT (1024)

#define DEBUG_TETRAHEDRON
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

template<unsigned Order, typename T>
struct Cache_Table {
    constexpr static unsigned Size = 1 << Order;
    constexpr static unsigned Mask = Size - 1;
    struct Key_Value {
        unsigned k;
        T value;
    };

    Key_Value table[Size];

    Cache_Table() {
        clear();
    }

    void clear() {
        for (auto& kv : table) {
            kv.k = UINT_MAX;
            kv.value = T();
        }
    }

    T fetch_or_insert(unsigned key, std::function<T(unsigned key)> f) {
        auto off = key & Mask;
        assert(off < Size);

        auto& slot = table[off];
        if (slot.k == key) {
            return slot.value;
        }

        slot.k = key;
        slot.value = f(key);
        return slot.value;
    }
};

static float randf() {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

void Softbody_Simulation::initialize(sb::Config const& configuration) {
    auto o = configuration.seed_position;
#ifdef DEBUG_TETRAHEDRON
#if 1
    auto siz = Vec3(1, 1, 2);
    auto idx_root = add_particle(o, siz, 1);
    auto idx_t0 = add_particle(o + Vec3(-4,  8,  4), siz, 1);
    auto idx_t1 = add_particle(o + Vec3(+4,  8,  4), siz, 1);
    auto idx_t2 = add_particle(o + Vec3( 0,  8, -4), siz, 1);

    connect_particles(idx_root, idx_t0);
    connect_particles(idx_root, idx_t1);
    connect_particles(idx_root, idx_t2);
    connect_particles(idx_t0, idx_t1);
    connect_particles(idx_t1, idx_t2);
    connect_particles(idx_t0, idx_t2);
#else
    auto siz = Vec3(0.5, 0.5, 1);
    auto idx_root = add_particle(o, siz, 1);
    auto idx_t0 = add_particle(o + Vec3(0, 1.0, 0), siz, 1);
    auto idx_t1 = add_particle(o + Vec3(0, 2.0, 0), siz, 1);
    auto idx_t2 = add_particle(o + Vec3(0, 3.0, 0), siz, 1);
    auto x90 = glm::normalize(Quat(Vec3(0, 0, glm::radians(90.0f))));
    orientation[idx_root] = x90;
    orientation[idx_t0] = x90;
    orientation[idx_t1] = x90;
    orientation[idx_t2] = x90;

    connect_particles(idx_root, idx_t0);
    connect_particles(idx_t0, idx_t1);
    connect_particles(idx_t1, idx_t2);
#endif
#endif /* DEBUG_TETRAHEDRON */

    params = configuration;

    if (params.particle_count_limit > SIM_SIZE_LIMIT) {
        params.particle_count_limit = SIM_SIZE_LIMIT;
    }

    center_of_mass.resize(position.size());
}

void Softbody_Simulation::prediction(float dt) {
    predicted_position.resize(position.size());
    predicted_orientation.resize(orientation.size());
    center_of_mass.resize(position.size());
    for (unsigned i = 0; i < position.size(); i++) {
        // prediction step
#if 1
        auto external_forces = Vec3(0, -1, 0);
#else
        auto external_forces = Vec3(0, 0, 0);
#endif
        auto v = velocity[i] + dt * (1 / mass_of_particle(i)) * external_forces;
        auto pos = position[i] + dt * v;

        auto ang_v = glm::length(angular_velocity[i]);
        Quat q;
        if (ang_v < 0.01) {
            // Angular velocity is too small; for stability reasons we keep
            // the old orientation
            q = orientation[i];
        } else {
            q = Quat(glm::cos(ang_v * dt / 2.0f), angular_velocity[i] / ang_v * glm::sin(ang_v * dt / 2.0f));
        }

        predicted_position[i] = pos;
        predicted_orientation[i] = q;
    }
}

#define NUMBER_OF_CLUSTERS(idx) (edges[(idx)].size() + 1)

void Softbody_Simulation::constraint_resolution(float dt) {
    constexpr unsigned SOLVER_ITERATIONS = 5;

    for (auto iter = 0ul; iter < SOLVER_ITERATIONS; iter++) {
        // shape matching constraint
        for (unsigned i = 0; i < position.size(); i++) {
            std::array<unsigned, 1> me{ i };
            auto& neighbors = edges[i];
            auto neighbors_and_me = iterator_union(neighbors.begin(), neighbors.end(), me.begin(), me.end());

            auto M = std::accumulate(
                neighbors.begin(), neighbors.end(),
                mass_of_particle(i),
                [&](float acc, unsigned idx) {
                    return acc + mass_of_particle(idx);
                }
            );

            assert(M != 0);

            // bind pose center of mass
            auto com0 = std::accumulate(
                neighbors.begin(), neighbors.end(),
                bind_pose[i],
                [&](Vec3 const& acc, unsigned idx) {
                    return acc + bind_pose[idx];
                }
            ) / M;

            auto calc_A_0_i = [&](unsigned i) -> Mat3 {
                auto q_i = bind_pose[i] - com0;
                auto m_i = mass_of_particle(i);
                auto x2 = m_i * q_i.x * q_i.x;
                auto y2 = m_i * q_i.y * q_i.y;
                auto z2 = m_i * q_i.z * q_i.z;
                auto xy = m_i * q_i.x * q_i.y;
                auto xz = m_i * q_i.x * q_i.z;
                auto yz = m_i * q_i.y * q_i.z;
                auto col0 = Vec3(x2, xy, xz);
                auto col1 = Vec3(xy, y2, yz);
                auto col2 = Vec3(xz, yz, z2);

                return Mat3(col0, col1, col2);
            };

            Mat3 A_0 = std::accumulate(
                neighbors.begin(), neighbors.end(),
                calc_A_0_i(i),
                [&](auto acc, auto idx) { return acc + calc_A_0_i(idx); }
            );

            Mat3 invRest;

            if (glm::abs(glm::determinant(A_0)) > glm::epsilon<float>()) {
                invRest = glm::inverse(A_0);
            } else {
                invRest = Mat3(1.0f);
            }
            
            auto com_cur = std::accumulate(
                neighbors.begin(), neighbors.end(),
                predicted_position[i],
                [&](Vec3 const& acc, unsigned idx) {
                    return acc + predicted_position[idx];
                }
            ) / M;

            center_of_mass[i] = com_cur;

            auto calc_A_i = [&](unsigned i) -> Mat3 {
                auto m_i = mass_of_particle(i);
                auto q_i = bind_pose[i] - com0;
                auto p_i = m_i * (predicted_position[i] - com_cur);

                auto col0 = p_i * Vec3(q_i.x, q_i.x, q_i.x);
                auto col1 = p_i * Vec3(q_i.y, q_i.y, q_i.y);
                auto col2 = p_i * Vec3(q_i.z, q_i.z, q_i.z);

                return Mat3(col0, col1, col2);
            };


            Mat3 A = std::accumulate(
                neighbors.begin(), neighbors.end(),
                calc_A_i(i),
                [&](auto acc, auto idx) { return acc + calc_A_i(idx); }
            ) * invRest;

            Quat R = predicted_orientation[i];
            mueller_rotation_extraction(A, R);

            float const stiffness = 1;

            for (auto idx : neighbors_and_me) {
                auto goal = com_cur + R * (bind_pose[i] - com0);
                auto numClusters = NUMBER_OF_CLUSTERS(idx);
                auto correction = (goal - predicted_position[i]) * stiffness;
                predicted_position[idx] += (1.0f / (float)numClusters) * correction;
                goal_position[idx] = goal;
            }

            predicted_orientation[i] = R;
        }

        // distance constraint
        for (unsigned i = 0; i < position.size(); i++) {
            auto& neighbors = edges[i];

            auto w1 = 1 / mass_of_particle(i);
            for (auto j : neighbors) {
                auto w2 = 1 / mass_of_particle(j);
                auto w = w1 + w2;

                auto n = predicted_position[j] - predicted_position[i];
                auto d = glm::length(n);
                n = glm::normalize(n);
                auto restLength = glm::length(bind_pose[j] - bind_pose[i]);

                auto stiffness = 1.0f;
                auto corr = stiffness * n * (d - restLength) / w;

                predicted_position[i] +=  w1 * corr;
                predicted_position[j] += -w2 * corr;
            }
        }
    }
}

void Softbody_Simulation::integration(float dt) {
    for (unsigned i = 0; i < position.size(); i++) {
        velocity[i] = (predicted_position[i] - position[i]) / dt;
        position[i] = predicted_position[i];

        auto r_tmp = predicted_orientation[i] * glm::conjugate(orientation[i]);
        auto r = (r_tmp.w < 0) ? -r_tmp : r_tmp;
        auto q_angle = glm::angle(r);
        if (glm::abs(q_angle) < 0.1) {
            angular_velocity[i] = Vec3(0, 0, 0);
        } else {
            angular_velocity[i] = glm::axis(r) * q_angle / dt;
        }

        orientation[i] = predicted_orientation[i];
        // TODO(danielm): friction?
    }
}

unsigned Softbody_Simulation::add_particle(Vec3 const& p_pos, Vec3 const& p_size, float p_density) {
    assert(!assert_parallel);
    assert(p_density >= 0.0f && p_density <= 1.0f);
    Vec3 zero(0, 0, 0);
    unsigned const index = position.size();
    bind_pose.push_back(p_pos);
    position.push_back(p_pos);
    predicted_position.push_back(p_pos);
    velocity.push_back(zero);
    angular_velocity.push_back(zero);
    goal_position.push_back(p_pos);
    size.push_back(p_size);
    density.push_back(p_density);
    orientation.push_back(Quat(1.0f, 0.0f, 0.0f, 0.0f));
    //age.push_back(0);
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

            s->prediction(phdt);
            s->constraint_resolution(phdt);
            s->integration(phdt);

            if (s->time_accumulator > 8 * PHYSICS_STEP) {
                fprintf(stderr, "EXTREME LAG, ACC = %f\n", s->time_accumulator);
            }

            // The creation of new particles is deferred until after all the
            // groups have been simulated
            for (auto& def_func : s->deferred) {
                def_func();
            }
            s->deferred.clear();

            s->time_accumulator -= PHYSICS_STEP;
        }
    }
}
