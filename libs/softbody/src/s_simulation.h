// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: softbody simulation library
//

#pragma once

#include "common.h"
#include "softbody.h"
#include "s_ext.h"

class ICompute_Backend;

class Softbody_Simulation : public sb::ISoftbody_Simulation, public IParticle_Manager, public IParticle_Manager_Deferred, public ILogger {
public:
    Softbody_Simulation(sb::Config const& configuration, sb::Debug_Proc dbg_msg_cb, void* dbg_msg_user);

    void add_particles(int N, glm::vec4 const* positions) override;
    void add_connections(int N, long long* pairs) override;

    void set_light_source_position(glm::vec3 const& pos) override;

    void step(float delta_time) override;

    sb::Unique_Ptr<sb::ISingle_Step_State> begin_single_step() override;
    sb::Unique_Ptr<sb::Particle_Iterator> get_particles() override;
    sb::Unique_Ptr<sb::Particle_Iterator> get_particles_with_goal_positions() override;
    sb::Unique_Ptr<sb::Particle_Iterator> get_particles_with_predicted_positions() override;
    sb::Unique_Ptr<sb::Particle_Iterator> get_centers_of_masses() override;
    sb::Unique_Ptr<sb::Relation_Iterator> get_connections() override;
    sb::Unique_Ptr<sb::Relation_Iterator> get_predicted_connections() override;

    sb::Unique_Ptr<sb::Particle_Iterator> get_particles_in_bind_pose() override;
    sb::Unique_Ptr<sb::Relation_Iterator> get_connections_in_bind_pose() override;

    void velocity_damping(float dt);
    void prediction(float dt);
    void constraint_resolution(float dt);
    void integration(float dt);

    Vector<Collision_Constraint> generate_collision_constraints();

    bool add_collider(
            Collider_Handle& out_handle,
            sb::sdf::ast::Expression<float>* sdf_expression,
            sb::sdf::ast::Sample_Point* sample_point) override;
    bool remove_collider(Collider_Handle handle) override;
    void collider_changed(Collider_Handle handle) override;
    bool add_collider(Collider_Handle &out_handle, sb::Mesh_Collider const *mesh) override;
    bool update_transform(Collider_Handle handle, glm::mat4 const &transform) override;

    size_t particle_count() const { return s.position.size(); }

    void pump_deferred_requests();

    // manual control
    float get_phdt();
    void do_one_iteration_of_distance_constraint_resolution(float phdt);
    void do_one_iteration_of_fixed_constraint_resolution(float phdt);
    void do_one_iteration_of_collision_constraint_resolution(float phdt);

    // Add a new particle to the simulation
    // This must only be called in the initial state.
    index_t add_init_particle(Vec3 const& p_pos, Vec3 const& p_size, float p_density) override;

    // Connect two particles. This has two effects to be exact:
    // - There will be a distance constraint between these two particles
    // - Both particles will be added to the other particle's cluster
    void connect_particles(index_t a, index_t b) override;

    virtual void add_fixed_constraint(unsigned count, index_t* pidx) override;

    bool save_image(sb::ISerializer* serializer) override;
    bool load_image(sb::IDeserializer* deserializer) override;

    void set_debug_visualizer(sb::IDebug_Visualizer *pVisualizer) override;

    ISimulation_Extension* create_extension(sb::Extension ext, sb::Config const& config);

    // Add a new particle to the simulation
    // This can be called when the system state has been already mutated, but
    // the caller must tell us who is this particle connected to.
    index_t add_particle(Vec3 const& p_pos, Vec3 const& p_size, float p_density, index_t parent) override;
    float mass_of_particle(index_t i);

    void invalidate_particle_cache(index_t pidx);
    void invalidate_particle_cache() override;

    sb::IPlant_Simulation* get_extension_plant_simulation() override;

    void debug_message_callback(sb::Debug_Proc callback, void* user) override;
    void log(sb::Debug_Message_Source s, sb::Debug_Message_Type t, sb::Debug_Message_Severity l, char const* fmt, ...) override;

    sb::Debug_Proc debugproc = nullptr;
    void* debugproc_user = nullptr;

    sb::Unique_Ptr<ICompute_Backend> compute;
    sb::Unique_Ptr<ISimulation_Extension> ext;

    System_State s;

    float time_accumulator = 0.0f;

    // Is the system currently being mutated? (OBSOLETE)
    bool assert_parallel;
    // Are we still in the initial state, that is, before the first call to step()
    bool assert_init;
    sb::Config params;
    sb::IDebug_Visualizer *m_pVisualizer = nullptr;

    // Stores functions whose execution has been deferred until after the parallelized
    // part
    Mutex deferred_lock;
    Vector<Fun<void(IParticle_Manager*, System_State& s)>> deferred;
    virtual void defer(std::function<void(IParticle_Manager* pman, System_State& s)> const& f) override;
};