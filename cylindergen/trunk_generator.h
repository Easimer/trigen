// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: wood trunk mesh generation declarations
//

#pragma once
#include <trigen/catmull_rom.h>
#include <trigen/linear_math.h>
#include "meshbuilder.h"

// Catmull_Rom<lm::Vector4>

Mesh_Builder::Optimized_Mesh MeshFromSpline(Catmull_Rom<lm::Vector4> const& cr);

