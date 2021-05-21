// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: iterators
//

#include "stdafx.h"
#include "softbody.h"
#include "s_simulation.h"
#include "m_utils.h"
#include "s_iterators.h"

sb::Unique_Ptr<sb::Particle_Iterator> Softbody_Simulation::get_particles() {
    auto pcg = [&]() { return s.position.size(); };
    auto pf = MAKE_PARTICLE_FACTORY(position);

    return std::make_unique<Particle_Iterator>(pcg, pf);
}

sb::Unique_Ptr<sb::Particle_Iterator> Softbody_Simulation::get_particles_with_goal_positions() {
    auto pcg = [&]() { return s.position.size(); };
    auto pf = MAKE_PARTICLE_FACTORY(goal_position);

    return std::make_unique<Particle_Iterator>(pcg, pf);
}

sb::Unique_Ptr<sb::Particle_Iterator> Softbody_Simulation::get_particles_with_predicted_positions() {
    auto pcg = [&]() { return s.position.size(); };
    auto pf = MAKE_PARTICLE_FACTORY(predicted_position);

    return std::make_unique<Particle_Iterator>(pcg, pf);
}

sb::Unique_Ptr<sb::Particle_Iterator> Softbody_Simulation::get_centers_of_masses() {
    auto pcg = [&]() { return s.position.size(); };
    auto pf = MAKE_PARTICLE_FACTORY(center_of_mass);

    return std::make_unique<Particle_Iterator>(pcg, pf);
}

sb::Unique_Ptr<sb::Relation_Iterator> Softbody_Simulation::get_connections() {
    auto get_map = [&]() -> decltype(s.edges)& { return s.edges; };
    auto make_relation = [&](index_t lhs, index_t rhs) {
        return sb::Relation {
            lhs,
            s.position[lhs],
            rhs,
            s.position[rhs],
        };
    };

    return std::make_unique<One_To_Many_Relation_Iterator>(get_map, make_relation);
}

sb::Unique_Ptr<sb::Relation_Iterator> Softbody_Simulation::get_predicted_connections() {
    auto get_map = [&]() -> decltype(s.edges)& { return s.edges; };
    auto make_relation = [&](index_t lhs, index_t rhs) {
        return sb::Relation {
            lhs,
            s.predicted_position[lhs],
            rhs,
            s.predicted_position[rhs],
        };
    };

    return std::make_unique<One_To_Many_Relation_Iterator>(get_map, make_relation);
}

sb::Unique_Ptr<sb::Particle_Iterator> Softbody_Simulation::get_particles_in_bind_pose() {
    auto pcg = [&]() { return s.position.size(); };
    auto pf = MAKE_PARTICLE_FACTORY(bind_pose);

    return std::make_unique<Particle_Iterator>(pcg, pf);
}

sb::Unique_Ptr<sb::Relation_Iterator> Softbody_Simulation::get_connections_in_bind_pose() {
    auto get_map = [&]() -> decltype(s.edges)& { return s.edges; };
    auto make_relation = [&](index_t lhs, index_t rhs) {
        return sb::Relation {
            lhs,
            s.bind_pose[lhs],
            rhs,
            s.bind_pose[rhs],
        };
    };

    return std::make_unique<One_To_Many_Relation_Iterator>(get_map, make_relation);
}
