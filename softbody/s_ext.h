// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: simulation extensions
//

#pragma once

#include "common.h"
#include "softbody.h"
#include <array>

class IParticle_Manager {
public:
    virtual unsigned add_init_particle(Vec3 const& p_pos, Vec3 const& p_size, float p_density) = 0;
    virtual void connect_particles(unsigned a, unsigned b) = 0;

    virtual unsigned add_particle(Vec3 const& p_pos, Vec3 const& p_size, float p_density, unsigned parent) = 0;

    virtual void add_fixed_constraint(unsigned count, unsigned* pidx) = 0;
};

class IParticle_Manager_Deferred {
public:
    virtual void defer(std::function<void(IParticle_Manager* pman, System_State& s)> const& f) = 0;
};

class ISimulation_Extension {
public:
    virtual ~ISimulation_Extension() {}
#define SIMEXT_CALLBACK(name) virtual void name (IParticle_Manager_Deferred*, System_State&, float dt) {}
    SIMEXT_CALLBACK(init)
    SIMEXT_CALLBACK(pre_prediction)
    SIMEXT_CALLBACK(post_prediction)
    SIMEXT_CALLBACK(pre_constraint)
    SIMEXT_CALLBACK(post_constraint)
    SIMEXT_CALLBACK(pre_integration)
    SIMEXT_CALLBACK(post_integration)
};

sb::Unique_Ptr<ISimulation_Extension> Create_Extension_Plant_Simulation(sb::Extension kind, sb::Config const& params);
sb::Unique_Ptr<ISimulation_Extension> Create_Extension_Cloth_Demo(sb::Extension kind, sb::Config const& params);

inline sb::Unique_Ptr<ISimulation_Extension> Create_Extension(sb::Extension kind, sb::Config const& params) {
    switch (kind) {
    case sb::Extension::Plant_Simulation: return Create_Extension_Plant_Simulation(kind, params);
    case sb::Extension::Debug_Cloth: return Create_Extension_Cloth_Demo(kind, params);
    default: return std::make_unique<ISimulation_Extension>();
    }
}
