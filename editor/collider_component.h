// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <memory>
#include <trigen.hpp>

#include "world.h"
#include "mesh_collider.h"

struct Collider_Component {
	std::unique_ptr<IMesh_Collider> mesh_collider;
	std::unordered_map<Entity_Handle, trigen::Collider> sb_handles;
};

