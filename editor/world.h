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

using Entity_Handle = std::make_signed<size_t>::type;

#include "transform_component.h"
#include "plant_component.h"
#include "collider_component.h"
#include "mesh_render_component.h"

class World {
public:
	World() {
	}

	Entity_Handle createEntity();

	template<typename T, typename ...Args>
	T *attachComponent(Entity_Handle ent, Args... args) {
		T component(args...);

		getMapForComponent<T>().emplace(std::make_pair(ent, std::move(component)));

		return &getMapForComponent<T>().at(ent);
	}

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

	template<>
	std::unordered_map<Entity_Handle, Mesh_Render_Component> &getMapForComponent<Mesh_Render_Component>() {
		return _c_mesh_render;
	}

	template<>
	std::unordered_map<Entity_Handle, Transform_Component> &getMapForComponent<Transform_Component>() {
		return _c_transform;
	}

private:
    struct Entity {
    };

	std::vector<std::optional<Entity>> _entities;

	std::unordered_map<Entity_Handle, Transform_Component> _c_transform;
	std::unordered_map<Entity_Handle, Collider_Component> _c_collider;
	std::unordered_map<Entity_Handle, Plant_Component> _c_plant;
	std::unordered_map<Entity_Handle, Mesh_Render_Component> _c_mesh_render;
};
