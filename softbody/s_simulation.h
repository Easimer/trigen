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
    Vector<Vec3> bind_pose;
    // Position in the previous frame
    Vector<Vec3> position;
    // Position in the current frame
    Vector<Vec3> predicted_position;
    Vector<Vec3> goal_position;

    // Particle velocities
    Vector<Vec3> velocity;
    // Particle angular velocities
    Vector<Vec3> angular_velocity;

    // Particle sizes
    Vector<Vec3> size;

    // Particle orientations in the last frame
    Vector<Quat> orientation;
    // Particle orientations in the current frame
    Vector<Quat> predicted_orientation;

    // Particle densities
    Vector<float> density;
    // Particle ages
    //Vector<float> age;
    Map<unsigned, Vector<unsigned>> edges;
    Map<unsigned, unsigned> apical_child;
    Map<unsigned, unsigned> lateral_bud;

    // For debug visualization only
    Vector<Vec3> center_of_mass;


    bool assert_parallel = false;

    float time_accumulator = 0.0f;

    Vec3 light_source = Vec3(0, 0, 0);

    sb::Config params;

    // Stores functions whose execution has been deferred until after the parallelized
    // part
    Mutex deferred_lock;
    Vector<Fun<void()>> deferred;

    void initialize(sb::Config const& configuration);

    void prediction(float dt);
    void constraint_resolution(float dt);
    void integration(float dt);

    // manual control
    float get_phdt();
    void do_one_iteration_of_shape_matching_constraint_resolution(float phdt);
    void do_one_iteration_of_distance_constraint_resolution(float phdt);
    void do_one_iteration_of_fixed_constraint_resolution(float phdt);

private:
    unsigned add_particle(Vec3 const& p_pos, Vec3 const& p_size, float p_density);
    void connect_particles(unsigned a, unsigned b);
    float mass_of_particle(unsigned i);
};
