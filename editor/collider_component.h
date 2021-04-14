// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <memory>
#include <softbody.h>

#include "world.h"
#include "mesh_collider.h"

struct Collider_Component {
	std::unique_ptr<IMesh_Collider> mesh_collider;
	std::unordered_map<Entity_Handle, sb::ISoftbody_Simulation::Collider_Handle> sb_handles;
};

