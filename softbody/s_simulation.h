// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: softbody simulation library
//

#pragma once

#include "common.h"
#include "softbody.h"
#include "s_ext.h"

class ICompute_Backend;

class Softbody_Simulation : public sb::ISoftbody_Simulation, public IParticle_Manager {
public:
    Softbody_Simulation(sb::Config const& configuration);

    void set_light_source_position(glm::vec3 const& pos) override;

    void step(float delta_time) override;

    sb::Unique_Ptr<sb::ISingle_Step_State> begin_single_step() override;
    sb::Unique_Ptr<sb::Particle_Iterator> get_particles() override;
    sb::Unique_Ptr<sb::Particle_Iterator> get_particles_with_goal_positions() override;
    sb::Unique_Ptr<sb::Particle_Iterator> get_particles_with_predicted_positions() override;
    sb::Unique_Ptr<sb::Particle_Iterator> get_centers_of_masses() override;
    sb::Unique_Ptr<sb::Relation_Iterator> get_apical_relations() override;
    sb::Unique_Ptr<sb::Relation_Iterator> get_lateral_relations() override;
    sb::Unique_Ptr<sb::Relation_Iterator> get_connections() override;
    sb::Unique_Ptr<sb::Relation_Iterator> get_predicted_connections() override;

    void prediction(float dt);
    void constraint_resolution(float dt);
    void integration(float dt);

    size_t particle_count() const { return s.position.size(); }

    // manual control
    float get_phdt();
    void do_one_iteration_of_distance_constraint_resolution(float phdt);
    void do_one_iteration_of_fixed_constraint_resolution(float phdt);

    unsigned add_particle(Vec3 const& p_pos, Vec3 const& p_size, float p_density) override;
    void connect_particles(unsigned a, unsigned b) override;
    float mass_of_particle(unsigned i);

    void invalidate_particle_cache(unsigned pidx);

    sb::Unique_Ptr<ICompute_Backend> compute;

    System_State s;

    float time_accumulator = 0.0f;

    bool assert_parallel;
    Vec3 light_source = Vec3(0, 0, 0);
    sb::Config params;

    // Stores functions whose execution has been deferred until after the parallelized
    // part
    Mutex deferred_lock;
    Vector<Fun<void()>> deferred;
};
