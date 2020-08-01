// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: plant simulation
//

#include "stdafx.h"
#include "common.h"
#include "softbody.h"
#include "s_ext.h"

class Plant_Simulation : public ISimulation_Extension {
    void post_prediction(IParticle_Manager* pman, System_State& s) override {
    }
};

sb::Unique_Ptr<ISimulation_Extension> Create_Extension_Plant_Simulation(sb::Extension kind) {
    return std::make_unique<Plant_Simulation>();
}
