// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: softbody simulation library
//

#pragma once

#include <glm/vec3.hpp>

struct Softbody_Simulation;

namespace sb {
    struct Config {
        glm::vec3 seed_position;
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
    };

    struct Arrow {
        glm::vec3 origin, direction;
    };

    template<typename T>
    class Iterator {
    public:
        virtual void release() = 0;

        virtual void step() = 0;
        virtual bool ended() const = 0;
        virtual T get() const = 0;
    };

    using Relation_Iterator = Iterator<Relation>;
    using Particle_Iterator = Iterator<Particle>;

    Softbody_Simulation* create_simulation(Config const& configuration);
    void destroy_simulation(Softbody_Simulation*);

    void set_light_source_position(Softbody_Simulation*, glm::vec3 const& pos);

    void step(Softbody_Simulation*, float delta_time);

    Particle_Iterator* get_particles(Softbody_Simulation*);
    Relation_Iterator* get_apical_relations(Softbody_Simulation*);
    Relation_Iterator* get_lateral_relations(Softbody_Simulation*);
}
