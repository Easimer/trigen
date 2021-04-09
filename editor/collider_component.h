// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <memory>
#include <softbody.h>

class IMesh_Collider {
public:
	virtual sb::Mesh_Collider *collider() = 0;
};

struct Collider_Component {
	std::unique_ptr<IMesh_Collider> mesh_collider;
	std::unordered_map<Plant_Component *, sb::ISoftbody_Simulation::Collider_Handle> sb_handles;
};

