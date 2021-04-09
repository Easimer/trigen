// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <list>
#include <memory>
#include <vector>
#include <optional>
#include <unordered_map>
#include <filament/Scene.h>

#include <utils/Entity.h>
#include <utils/EntityManager.h>

#include "world_plant.h"
#include "world_collider.h"

class World {
public:
	World(filament::Scene *scene) : _scene(scene) {
	}

	using Entity_Handle = std::make_signed<size_t>::type;

	Entity_Handle createEntity() {
		std::optional<Entity> *slot = nullptr;
		Entity_Handle ret;

		for (Entity_Handle i = 0; i < _entities.size(); i++) {
			if (!_entities[i].has_value()) {
				slot = &_entities[i];
				ret = i;
				break;
			}
		}

		if (slot == nullptr) {
			_entities.push_back({});
			slot = &_entities.back();
			ret = _entities.size() - 1;
		}

		*(*slot) = { utils::EntityManager::get().create() };

		return ret;
	}

	template<typename T, typename ...Args>
	T *attachComponent(Entity_Handle ent, Args... args) {
		T component(args...);

		getMapForComponent<T>().emplace(std::make_pair(ent, std::move(component)));

		return &getMapForComponent<T>().at(ent);
	}

protected:
	template<typename T>
	std::unordered_map<Entity_Handle, T> &getMapForComponent();

	template<>
	std::unordered_map<Entity_Handle, Collider_Component> &getMapForComponent<Collider_Component>() {
		return _c_collider;
	}

	template<>
	std::unordered_map<Entity_Handle, Plant_Component> &getMapForComponent<Plant_Component>() {
		return _c_plant;
	}

private:
	filament::Scene *_scene;

	struct Entity {
		utils::Entity entity;
	};

	std::vector<std::optional<Entity>> _entities;

	std::unordered_map<Entity_Handle, Collider_Component> _c_collider;
	std::unordered_map<Entity_Handle, Plant_Component> _c_plant;
};
