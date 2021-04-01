// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include "world_object.h"
#include <utils/EntityManager.h>
#include <string>
#include <softbody.h>
#include "world_collider.h"

class World_Plant : public World_Object {
public:
	~World_Plant() override = default;

	World_Plant() {
		_entity = utils::EntityManager::get().create();
	}

	char const *className() const override {
		return "plant";
	}

	void onObjectAdded(World_Object const *object) override {
		if (object->className() == colliderClassName) {
			auto collider = (World_Collider const *)object;
		}
	}

private:
	std::string const colliderClassName = WORLD_COLLIDER_CLASSNAME;
};