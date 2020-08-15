// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: serialization and deserialization
//

#pragma once

#include "softbody.h"
#include "common.h"

enum class Serialization_Result {
    OK = 0,
    Bad_Format,
    Bad_Version,
};

bool sim_save_image(System_State const& s, sb::ISerializer* serializer);
Serialization_Result sim_load_image(System_State& s, sb::IDeserializer* deserializer);
