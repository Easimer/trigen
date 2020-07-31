// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: softbody simulation
//

#include "stdafx.h"
#include "softbody.h"
#include "m_utils.h"
#include "l_iterators.h"
#include "s_simulation.h"
#include <cstdlib>
#include <array>
#include <glm/gtx/matrix_operation.hpp>
#include "s_compute_backend.h"

// #include <CL/sycl.hpp>
// namespace sycl = cl::sycl;

#define PHYSICS_STEP (1.0f / 25.0f)
#define TAU (PHYSICS_STEP)
#define SIM_SIZE_LIMIT (1024)
#define SOLVER_ITERATIONS (5)

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

Softbody_Simulation::Softbody_Simulation(sb::Config const& configuration)
    : assert_parallel(false) {
    auto o = configuration.seed_position;
#ifdef DEBUG_TETRAHEDRON
#if 0
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
    auto siz = Vec3(0.25, 0.25, 0.5);
    auto x90 = glm::normalize(Quat(Vec3(0, 0, glm::radians(90.0f))));
    auto idx_root = add_particle(o, siz, 1);
    s.orientation[idx_root] = x90;
    auto prev = idx_root;
    for (int i = 0; i < 1024; i++) {
        auto cur = add_particle(o + Vec3((i + 1) * 0.5f, 0, 0), siz, 1);
        s.orientation[cur] = x90;
        connect_particles(prev, cur);
        prev = cur;
    }
#endif
#else
    auto siz = Vec3(1, 1, 2);
    auto idx_root = add_particle(o, Vec3(0.25, 0.25, 0.50), 1);
#endif /* DEBUG_TETRAHEDRON */

    params = configuration;

    if (params.particle_count_limit > SIM_SIZE_LIMIT) {
        params.particle_count_limit = SIM_SIZE_LIMIT;
    }

    s.center_of_mass.resize(particle_count());

    compute = Make_Compute_Backend();
    ext = Create_Extension(params.ext);
}

void Softbody_Simulation::prediction(float dt) {
    s.predicted_position.resize(particle_count());
    s.predicted_orientation.resize(s.orientation.size());
    s.center_of_mass.resize(particle_count());

    ext->pre_prediction(this, s);

    for (unsigned i = 0; i < particle_count(); i++) {
        // prediction step
#if 1
        auto external_forces = Vec3(0, -10, 0);
#else
        auto external_forces = Vec3(0, 0, 0);
#endif
        auto v = s.velocity[i] + dt * (1 / mass_of_particle(i)) * external_forces;
        auto pos = s.position[i] + dt * v;

        auto ang_v = glm::length(s.angular_velocity[i]);
        Quat q;
        if (ang_v < 0.01) {
            // Angular velocity is too small; for stability reasons we keep
            // the old orientation
            q = s.orientation[i];
        } else {
            q = Quat(glm::cos(ang_v * dt / 2.0f), s.angular_velocity[i] / ang_v * glm::sin(ang_v * dt / 2.0f));
        }

        s.predicted_position[i] = pos;
        s.predicted_orientation[i] = q;
    }

    ext->post_prediction(this, s);
}

#define NUMBER_OF_CLUSTERS(idx) (s.edges[(idx)].size() + 1)

float Softbody_Simulation::get_phdt() {
    return PHYSICS_STEP;
}

void Softbody_Simulation::do_one_iteration_of_distance_constraint_resolution(float phdt) {
    for (unsigned i = 0; i < particle_count(); i++) {
        auto& neighbors = s.edges[i];

        auto w1 = 1 / mass_of_particle(i);
        for (auto j : neighbors) {
            auto w2 = 1 / mass_of_particle(j);
            auto w = w1 + w2;

            auto n = s.predicted_position[j] - s.predicted_position[i];
            auto d = glm::length(n);
            n = glm::normalize(n);
            auto restLength = glm::length(s.bind_pose[j] - s.bind_pose[i]);

            auto stiffness = params.stiffness;
            auto corr = stiffness * n * (d - restLength) / w;

            s.predicted_position[i] += w1 * corr;
            s.predicted_position[j] += -w2 * corr;
        }
    }
}

void Softbody_Simulation::do_one_iteration_of_fixed_constraint_resolution(float phdt) {
    // force particles to stay in their bind pose
    auto particles = { 0 };

    for (auto i : particles) {
        s.predicted_position[i] = Vec3();
    }
}

void Softbody_Simulation::constraint_resolution(float dt) {
    ext->pre_constraint(this, s);
    compute->begin_new_frame(s);

    for (auto iter = 0ul; iter < SOLVER_ITERATIONS; iter++) {
        compute->do_one_iteration_of_shape_matching_constraint_resolution(s, dt);
        do_one_iteration_of_distance_constraint_resolution(dt);
        do_one_iteration_of_fixed_constraint_resolution(dt);
    }

    ext->post_constraint(this, s);
}

void Softbody_Simulation::integration(float dt) {
    ext->pre_integration(this, s);

    for (unsigned i = 0; i < particle_count(); i++) {
        s.velocity[i] = (s.predicted_position[i] - s.position[i]) / dt;
        s.position[i] = s.predicted_position[i];

        auto r_tmp = s.predicted_orientation[i] * glm::conjugate(s.orientation[i]);
        auto r = (r_tmp.w < 0) ? -r_tmp : r_tmp;
        auto q_angle = glm::angle(r);
        if (glm::abs(q_angle) < 0.1) {
            s.angular_velocity[i] = Vec3(0, 0, 0);
        } else {
            s.angular_velocity[i] = glm::axis(r) * q_angle / dt;
        }

        s.orientation[i] = s.predicted_orientation[i];
        // TODO(danielm): friction?
    }

    ext->post_integration(this, s);
}

unsigned Softbody_Simulation::add_particle(Vec3 const& p_pos, Vec3 const& p_size, float p_density) {
    assert(!assert_parallel);
    assert(p_density >= 0.0f && p_density <= 1.0f);
    Vec3 zero(0, 0, 0);
    unsigned const index = particle_count();
    s.bind_pose.push_back(p_pos);
    s.position.push_back(p_pos);
    s.predicted_position.push_back(p_pos);
    s.velocity.push_back(zero);
    s.angular_velocity.push_back(zero);
    s.goal_position.push_back(p_pos);
    s.size.push_back(p_size);
    s.density.push_back(p_density);
    s.orientation.push_back(Quat(1.0f, 0.0f, 0.0f, 0.0f));
    //age.push_back(0);
    s.edges[index] = {};

    invalidate_particle_cache(index);

    return index;
}

void Softbody_Simulation::connect_particles(unsigned a, unsigned b) {
    assert(!assert_parallel);
    assert(a < particle_count());
    assert(b < particle_count());

    s.edges[a].push_back(b);
    s.edges[b].push_back(a);

    invalidate_particle_cache(a);
    invalidate_particle_cache(b);
}

float Softbody_Simulation::mass_of_particle(unsigned i) {
    auto const d_i = s.density[i];
    auto const s_i = s.size[i];
    auto const m_i = (4.f / 3.f) * glm::pi<float>() * s_i.x * s_i.y * s_i.z * d_i;
    return m_i;
}

void Softbody_Simulation::invalidate_particle_cache(unsigned pidx) {
    auto& neighbors = s.edges[pidx];

    auto M = std::accumulate(
        neighbors.begin(), neighbors.end(),
        mass_of_particle(pidx),
        [&](float acc, unsigned idx) {
            return acc + mass_of_particle(idx);
        }
    );

    auto com0 = std::accumulate(
        neighbors.begin(), neighbors.end(),
        mass_of_particle(pidx) * s.bind_pose[pidx],
        [&](Vec3 const& acc, unsigned idx) {
            return acc + mass_of_particle(idx) * s.bind_pose[idx];
        }
    ) / M;

    auto calc_A_0_i = [&](unsigned i) -> Mat3 {
        auto q_i = s.bind_pose[i] - com0;
        auto m_i = mass_of_particle(i);

        return m_i * glm::outerProduct(q_i, q_i);
    };

    // A_qq
    Mat3 A_0 = std::accumulate(
        neighbors.begin(), neighbors.end(),
        calc_A_0_i(pidx),
        [&](auto acc, auto idx) { return acc + calc_A_0_i(idx); }
    );

    Mat3 invRest;

    if (glm::abs(glm::determinant(A_0)) > glm::epsilon<float>()) {
        invRest = glm::inverse(A_0);
    } else {
        invRest = Mat3(1.0f);
    }

    size_t i = pidx;
    if (s.bind_pose_center_of_mass.size() <= i) {
        s.bind_pose_center_of_mass.resize(i + 1);
    }

    if (s.bind_pose_inverse_bind_pose.size() <= i) {
        s.bind_pose_inverse_bind_pose.resize(i + 1);
    }

    s.bind_pose_center_of_mass[i] = com0;
    s.bind_pose_inverse_bind_pose[i] = invRest;
}

sb::Unique_Ptr<sb::ISoftbody_Simulation> sb::create_simulation(Config const& configuration) {
    return std::make_unique<Softbody_Simulation>(configuration);
}

void Softbody_Simulation::set_light_source_position(Vec3 const& pos) {
    light_source = pos;
}

void Softbody_Simulation::step(float delta_time) {
    time_accumulator += delta_time;

    if (time_accumulator > PHYSICS_STEP) {
        auto phdt = PHYSICS_STEP;

        prediction(phdt);
        constraint_resolution(phdt);
        integration(phdt);

        if (time_accumulator > 8 * PHYSICS_STEP) {
            fprintf(stderr, "sb: warning: extreme lag, acc = %f\n", time_accumulator);
        }

        for (auto& def_func : deferred) {
            def_func();
        }
        deferred.clear();

        time_accumulator -= phdt;
    }
}

class Single_Step_State : public sb::ISingle_Step_State {
public:
    Single_Step_State(Softbody_Simulation* sim)
        : sim(sim), constraint_iteration(0), state(PREDICTION) {}

    ~Single_Step_State() {
        // we must bring the simulation into a valid state first
        while (state != Single_Step_State::PREDICTION) {
            step();
        }
    }

    void step() override {
        switch (state) {
        case Single_Step_State::PREDICTION:
        {
            sim->prediction(PHYSICS_STEP);
            state = Single_Step_State::CONSTRAINT_SHAPE_MATCH;
            break;
        }
        case Single_Step_State::CONSTRAINT_SHAPE_MATCH:
        {
            sim->compute->begin_new_frame(sim->s);
            sim->compute->do_one_iteration_of_shape_matching_constraint_resolution(sim->s, PHYSICS_STEP);
            state = Single_Step_State::CONSTRAINT_FIXED;
            break;
        }
        case Single_Step_State::CONSTRAINT_DISTANCE:
        {
            sim->do_one_iteration_of_distance_constraint_resolution(PHYSICS_STEP);
            if (constraint_iteration < SOLVER_ITERATIONS) {
                state = Single_Step_State::CONSTRAINT_SHAPE_MATCH;
                constraint_iteration++;
            } else {
                state = Single_Step_State::INTEGRATION;
                constraint_iteration = 0;
            }
            break;
        }
        case Single_Step_State::CONSTRAINT_FIXED:
        {
            sim->do_one_iteration_of_fixed_constraint_resolution(PHYSICS_STEP);
            state = Single_Step_State::CONSTRAINT_DISTANCE;
            break;
        }
        case Single_Step_State::INTEGRATION:
        {
            sim->integration(PHYSICS_STEP);
            state = Single_Step_State::PREDICTION;
            break;
        }
        }
    }

    void get_state_description(unsigned length, char* buffer) override {
        if (buffer != NULL && length > 0) {
            length = length - 1;
            switch (state) {
            case Single_Step_State::PREDICTION:
            {
                snprintf(buffer, length, "Before prediction step");
                break;
            }
            case Single_Step_State::CONSTRAINT_SHAPE_MATCH:
            {
                snprintf(buffer, length, "Before shape matching constraint resolution (iter=%d)", constraint_iteration);
                break;
            }
            case Single_Step_State::CONSTRAINT_DISTANCE:
            {
                snprintf(buffer, length, "Before distance constraint resolution (iter=%d)", constraint_iteration);
                break;
            }
            case Single_Step_State::CONSTRAINT_FIXED:
            {
                snprintf(buffer, length, "Before fixed position constraint resolution (iter=%d)", constraint_iteration);
                break;
            }
            case Single_Step_State::INTEGRATION:
            {
                snprintf(buffer, length, "Pre-integration");
                break;
            }
            }
            buffer[length] = '\0';
        }
    }

private:
    Softbody_Simulation* sim;
    int constraint_iteration;

    enum State {
        PREDICTION,
        CONSTRAINT_FIXED,
        CONSTRAINT_SHAPE_MATCH,
        CONSTRAINT_DISTANCE,
        INTEGRATION,
    } state;
};

sb::Unique_Ptr<sb::ISingle_Step_State> Softbody_Simulation::begin_single_step() {
    return std::make_unique<Single_Step_State>(this);
}
