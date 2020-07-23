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
#define TAU (PHYSICS_STEP)
#define SIM_SIZE_LIMIT (1024)

#define DEBUG_TETRAHEDRON
#ifdef DEBUG_TETRAHEDRON
#define DISABLE_GROWTH
#define DISABLE_PHOTOTROPISM
#endif /* defined(DEBUG_TETRAHEDRON) */

#define MULLER_2005 (0)
#define MULLER_2011 (1)

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

    if (params.particle_count_limit > SIM_SIZE_LIMIT) {
        params.particle_count_limit = SIM_SIZE_LIMIT;
    }
}

void Softbody_Simulation::predict_positions(float dt) {
    predicted_position.resize(position.size());
    for (unsigned i = 0; i < position.size(); i++) {
        auto pos = position[i];
        auto goal = goal_position[i];

        auto alpha = dt / TAU;
        // auto alpha = 1.0f;
        auto external_forces = Vec3(0, -1, 0);

        /*
        if (pos.y < 0) {
            external_forces += 4.0f * Vec3(0, -pos.y, 0);
        }
        */

        auto a = dt * external_forces / mass_of_particle(i);
        auto v = alpha * (goal - pos) / dt;
        velocity[i] = velocity[i] + alpha * (goal - pos) / dt + dt * external_forces / mass_of_particle(i);
        predicted_position[i] = position[i] + dt * velocity[i];

        if (goal.x - pos.x != 0) {
            auto t = (predicted_position[i].x - pos.x) / (goal.x - pos.x);
            if (t > 1) {
                printf("PARTICLE %u OVERSHOOT t=%f\n", i, t);
            }
        } else {
            auto t = glm::length(predicted_position[i] - pos);
            if (t > 1) {
                printf("PARTICLE %u OVERSHOOT t=%f\n", i, t);
            }
        }
    }
}

void Softbody_Simulation::commit_predicted_positions() {
    for (unsigned i = 0; i < position.size(); i++) {
        position[i] = predicted_position[i];
    }

    predicted_position.clear();
}

void Softbody_Simulation::simulate_group(unsigned pidx, float dt) {
    // Shape Matching
    // Mueller, et al 2005; 3.3

    Cache_Table<3, float> particle_mass_cache;

    auto get_mass = [&](unsigned idx) -> float {
        return particle_mass_cache.fetch_or_insert(idx, [this](unsigned idx) { return mass_of_particle(idx); });
    };

    // sum of masses
    auto& neighbors = edges[pidx];
    auto total_mass = std::accumulate(
        neighbors.begin(), neighbors.end(),
        // mass_of_particle(pidx),
        get_mass(pidx),
        [&](float acc, unsigned idx) -> float {
            // return acc + mass_of_particle(idx);
            return acc + get_mass(idx);
        }
    );
    // center of mass of the original configuration
    auto com0 = std::accumulate(
        neighbors.begin(), neighbors.end(),
        // mass_of_particle(pidx) * position[pidx],
        get_mass(pidx) * position[pidx],
        [&](Vec3 const& acc, unsigned idx) -> Vec3 {
            // return acc + mass_of_particle(idx) * position[idx];
            return acc + get_mass(idx) * position[idx];
        }
    ) / total_mass;
    
    // CoM of the current configuration
    auto com1 = std::accumulate(
        neighbors.begin(), neighbors.end(),
        // mass_of_particle(pidx) * predicted_position[pidx],
        get_mass(pidx) * predicted_position[pidx],
        [&](Vec3 const& acc, unsigned idx) -> Vec3 {
            // return acc + mass_of_particle(idx) * predicted_position[idx];
            return acc + get_mass(idx) * predicted_position[idx];
        }
    ) / total_mass;

    auto calc_p = [=](unsigned idx) {
        return position[idx] - com0;
    };

    auto calc_q = [=](unsigned idx) {
        return predicted_position[idx] - com1;
    };

    auto A_pq = std::accumulate(
        neighbors.begin(), neighbors.end(),
        get_mass(pidx) * glm::outerProduct(calc_p(pidx), calc_q(pidx)),
        [=](Mat3 const& acc, unsigned idx) -> Mat3 {
            return acc + get_mass(idx) * glm::outerProduct(calc_p(idx), calc_q(idx));
        }
    );

    auto A_qq = glm::inverse(std::accumulate(
        neighbors.begin(), neighbors.end(),
        get_mass(pidx) * glm::outerProduct(calc_q(pidx), calc_q(pidx)),
        [=](Mat3 const& acc, unsigned idx) -> Mat3 {
            return acc + get_mass(idx) * glm::outerProduct(calc_q(idx), calc_q(idx));
        }
    ));

    // auto R = polar_decompose_r(A_pq);
    auto At_pq__A_pq = glm::transpose(A_pq) * A_pq;
    auto R = polar_decompose_r(At_pq__A_pq);
    

    goal_position[pidx] = R * (position[pidx] - com0) + com1;
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
            for (auto i : range(0, s->position.size())) {
                s->simulate_group(i, phdt);
            }
            s->commit_predicted_positions();

            // s->position[0] = p0;

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
