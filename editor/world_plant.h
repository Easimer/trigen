// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include "world_object.h"
#include <utils/EntityManager.h>

class World_Plant : public World_Object {
public:
	~World_Plant() override = default;

	World_Plant() {
		_entity = utils::EntityManager::get().create();
	}
};