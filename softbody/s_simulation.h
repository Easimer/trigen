// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: softbody simulation library
//

#pragma once

#include "common.h"

struct Particle_Group {
    unsigned owner;
    float owner_mass;
    Vector<unsigned> neighbors;
    Vector<float> masses;
    float W;
    Vec3 c, c_rest;
    Mat3 orient;
};

struct Softbody_Simulation {
    Vector<Vec3> position;
    Vector<Vec3> velocity;
    Vector<Vec3> angular_velocity;
    Vector<Vec3> rest_position;
    Vector<Vec3> goal_position;
    Vector<Vec3> center_of_mass;
    Vector<Vec3> rest_center_of_mass;
    Vector<Vec3> size;
    Vector<Quat> orientation;
    Vector<float> density;
    Map<unsigned, Vector<unsigned>> edges;
    Map<unsigned, unsigned> apical_child;
    Map<unsigned, unsigned> lateral_bud;

    Vector<Vec3> predicted_position;

    bool assert_parallel = false;

    float time_accumulator = 0.0f;

    Vec3 light_source = Vec3(0, 0, 0);

    // Stores functions whose execution has been deferred until after the parallelized
    // part
    Mutex deferred_lock;
    Vector<Fun<void()>> deferred;

    void initialize(sb::Config const& configuration);
    void predict_positions(float dt);
    void simulate_group(unsigned pidx, float dt);

private:
    unsigned add_particle(Vec3 const& p_pos, Vec3 const& p_size, float p_density);
    void connect_particles(unsigned a, unsigned b);
    float mass_of_particle(unsigned i);
    void calculate_orientation_matrix(Particle_Group* group);
};
