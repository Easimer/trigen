// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: softbody simulation library
//

#pragma once

#include <functional>
#include <memory>
#include <raymarching.h>
#include <glm/vec3.hpp>
#include <glm/gtc/quaternion.hpp>

namespace sb {
    template<typename T, typename Deleter = std::default_delete<T>>
    using Unique_Ptr = std::unique_ptr<T, Deleter>;
    template<typename T>
    using Shared_Ptr = std::shared_ptr<T>;

    enum class Extension {
        None = 0,
        Debug_Rope,
        Debug_Cloth,
        Plant_Simulation,
    };

    enum class Compute_Preference {
        None = 0,
        Reference,
        GPU,
    };

    struct Config {
        Extension ext;
        glm::vec3 seed_position;

        float density;                          // rho
        float attachment_strength;              // phi
        float surface_adaption_strength;        // tau
        float stiffness;                        // s
        float aging_rate;                       // t_s
        float phototropism_response_strength;   // eta
        float branching_probability;
        float branch_angle_variance;

        unsigned particle_count_limit;
    };

    struct Relation {
        size_t parent;
        glm::vec3 parent_position;
        size_t child;
        glm::vec3 child_position;
    };

    struct Particle {
        size_t id;
        glm::vec3 position;
        glm::quat orientation;
        glm::vec3 size;

        glm::vec3 start, end;
    };

    struct Arrow {
        glm::vec3 origin, direction;
    };

    template<typename T>
    class Iterator {
    public:
        virtual ~Iterator() {}

        virtual void step() = 0;
        virtual bool ended() const = 0;
        virtual T get() const = 0;
    };

    using Relation_Iterator = Iterator<Relation>;
    using Particle_Iterator = Iterator<Particle>;

    class ISingle_Step_State {
    public:
        virtual ~ISingle_Step_State() {}

        virtual void step() = 0;
        virtual void get_state_description(unsigned length, char* buffer) = 0;
    };

    using Signed_Distance_Function = sdf::Function;

    class ISoftbody_Simulation {
    public:
        virtual ~ISoftbody_Simulation() {}

        virtual void set_light_source_position(glm::vec3 const& pos) = 0;
        virtual void step(float delta_time) = 0;

        virtual Unique_Ptr<ISingle_Step_State> begin_single_step() = 0;

        virtual Unique_Ptr<Particle_Iterator> get_particles() = 0;
        virtual Unique_Ptr<Particle_Iterator> get_particles_with_goal_positions() = 0;
        virtual Unique_Ptr<Particle_Iterator> get_particles_with_predicted_positions() = 0;
        virtual Unique_Ptr<Particle_Iterator> get_centers_of_masses() = 0;
        virtual Unique_Ptr<Relation_Iterator> get_connections() = 0;
        virtual Unique_Ptr<Relation_Iterator> get_predicted_connections() = 0;

        virtual Unique_Ptr<Particle_Iterator> get_particles_in_bind_pose() = 0;
        virtual Unique_Ptr<Relation_Iterator> get_connections_in_bind_pose() = 0;

        using Collider_Handle = size_t;
        virtual Collider_Handle add_collider(Signed_Distance_Function const& sdf) = 0;
        virtual void remove_collider(Collider_Handle handle) = 0;
    };

    Unique_Ptr<ISoftbody_Simulation> create_simulation(Config const& configuration);
}
