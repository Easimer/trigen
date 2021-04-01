// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <utils/Entity.h>

class World;

class World_Object {
public:
	virtual ~World_Object() = default;

	virtual char const *className() const = 0;

	virtual void setWorld(World *world) final {
		_world = world;
	}

	virtual void onObjectAdded(World_Object const *other) {}
	virtual void onObjectRemoved(World_Object const *other) {}
protected:
	World *_world;
	utils::Entity _entity;
};