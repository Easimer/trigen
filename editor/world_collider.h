// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include "world_object.h"

#define WORLD_COLLIDER_CLASSNAME "collider"

class IMesh_Collider {
public:
	virtual sb::Mesh_Collider *collider() = 0;
};

class World_Collider : public World_Object {
public:
	~World_Collider() override = default;

	World_Collider(std::unique_ptr<IMesh_Collider> &&mca) : _mesh_collider(std::move(mca)) {
		_handle = 0;
	}

	char const *className() const override {
		return WORLD_COLLIDER_CLASSNAME;
	}

private:
	std::unique_ptr<IMesh_Collider> _mesh_collider;
	sb::ISoftbody_Simulation::Collider_Handle _handle;
};