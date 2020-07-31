// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: simulation extensions
//

#pragma once

#include "common.h"
#include "softbody.h"

class IParticle_Manager {
public:
    virtual unsigned add_particle(Vec3 const& p_pos, Vec3 const& p_size, float p_density) = 0;
    virtual void connect_particles(unsigned a, unsigned b) = 0;
};

class ISimulation_Extension {
public:
    virtual ~ISimulation_Extension() {}
#define SIMEXT_CALLBACK(name) virtual void name (IParticle_Manager*, System_State&) {}
    SIMEXT_CALLBACK(pre_prediction)
    SIMEXT_CALLBACK(post_prediction)
    SIMEXT_CALLBACK(pre_constraint)
    SIMEXT_CALLBACK(post_constraint)
    SIMEXT_CALLBACK(pre_integration)
    SIMEXT_CALLBACK(post_integration)
};

sb::Unique_Ptr<ISimulation_Extension> Create_Extension_Plant_Simulation(sb::Extension kind);

inline sb::Unique_Ptr<ISimulation_Extension> Create_Extension(sb::Extension kind) {
    switch (kind) {
    case sb::Extension::Plant_Simulation: return Create_Extension_Plant_Simulation(kind);
    default: return std::make_unique<ISimulation_Extension>();
    }
}
