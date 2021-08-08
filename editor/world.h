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

#define DEFINE_GETMAPFORCOMPONENT(typeName, dataMember) \
	template<> \
	std::unordered_map<Entity_Handle, typeName> &getMapForComponent<typeName>() { \
		return dataMember; \
	} \
\
	template<> \
	std::unordered_map<Entity_Handle, typeName> const &getMapForComponent<typeName>() const { \
		return dataMember; \
	}

#define FOREACH_COMPONENT(func) \
	func(Collider_Component, _c_collider) \
	func(Plant_Component, _c_plant) \
	func(Mesh_Render_Component, _c_mesh_render) \
	func(Transform_Component, _c_transform)

class World {
public:
	Entity_Handle createEntity();
	bool removeEntity(Entity_Handle handle);

	template<typename T, typename ...Args>
	T *attachComponent(Entity_Handle ent, Args... args) {
		T component(args...);

		getMapForComponent<T>().emplace(std::make_pair(ent, std::move(component)));

		return &getMapForComponent<T>().at(ent);
	}

	template<typename T>
	std::unordered_map<Entity_Handle, T> &getMapForComponent();

	template<typename T>
	std::unordered_map<Entity_Handle, T> const &getMapForComponent() const;

	FOREACH_COMPONENT(DEFINE_GETMAPFORCOMPONENT)

	int numEntities() const {
		return _entities.size();
	}

	bool exists(Entity_Handle ent) const;

protected:
    struct Entity {
    };

	std::vector<std::optional<Entity>> _entities;

	std::unordered_map<Entity_Handle, Transform_Component> _c_transform;
	std::unordered_map<Entity_Handle, Collider_Component> _c_collider;
	std::unordered_map<Entity_Handle, Plant_Component> _c_plant;
	std::unordered_map<Entity_Handle, Mesh_Render_Component> _c_mesh_render;
};
