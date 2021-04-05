// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <list>
#include <memory>
#include <filament/Scene.h>
#include "world_object.h"

class World {
public:
	World(filament::Scene *scene) : _scene(scene) {
	}

	template<typename T, typename ...Args>
	void createEntity(Args... args) {
		auto ent = std::make_unique<T>(args...);
		ent->setWorld(this);

		for (auto &object : _objects) {
			object->onObjectAdded(ent.get());
		}

		_objects.emplace_back(std::move(ent));
	}

private:
	std::list<std::unique_ptr<World_Object>> _objects;
	filament::Scene *_scene;
};
