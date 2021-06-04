// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: serialization and deserialization
//

#pragma once

#include "softbody.h"
#include "system_state.h"
#include "s_simulation.h"

enum class Serialization_Result {
    OK = 0,
    Bad_Format,
    Bad_Version,
};

bool sim_save_image(System_State const& s, sb::ISerializer* serializer, ISimulation_Extension* ext);
Serialization_Result sim_load_image(Softbody_Simulation* sim, System_State& s, sb::IDeserializer* deserializer, sb::Config const& config);
