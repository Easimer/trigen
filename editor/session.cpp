// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"
#include "session.h"
#include "world.h"
#include <glm/gtc/type_ptr.hpp>
#include <unordered_set>

#include <r_cmd/general.h>
#include <r_cmd/softbody.h>

Session::Session(char const *name) :
	_name(name),
	_renderParams{},
	_world() {
}

void Session::createPlant(sb::Config const &cfg) {
	auto ent = _world.createEntity();
	_world.attachComponent<Plant_Component>(ent, cfg);
}

void Session::onTick(float deltaTime) {
	auto &colliders = _world.getMapForComponent<Collider_Component>();
	auto &plants = _world.getMapForComponent<Plant_Component>();

	for (auto &colliderEnt : colliders) {
        // Check whether all colliders are added to all plant sims
		for (auto &plantEnt : plants) {
			if (colliderEnt.second.sb_handles.count(plantEnt.first) == 0) {
				sb::ISoftbody_Simulation::Collider_Handle collHandle;
				if (plantEnt.second._sim->add_collider(collHandle, colliderEnt.second.mesh_collider->collider())) {
					colliderEnt.second.sb_handles.emplace(std::make_pair(plantEnt.first, collHandle));
				}
			}
		}

		// Remove SB collider handles that reference deleted plant entities
		std::unordered_set<Entity_Handle> to_be_removed;
		for (auto &handle : colliderEnt.second.sb_handles) {
			if (plants.count(handle.first) == 0) {
				to_be_removed.insert(handle.first);
			}
		}

		for (auto &handle : to_be_removed) {
			colliderEnt.second.sb_handles.erase(handle);
		}
	}

	for (auto &kv : plants) {
		if (kv.second.isRunning) {
			kv.second._sim->step(deltaTime);
		}
	}
}

void Session::onRender(gfx::Render_Queue *rq) {
	auto &plants = _world.getMapForComponent<Plant_Component>();

	gfx::allocate_command_and_initialize<Render_Grid>(rq);

	for (auto &kv : plants) {
        gfx::allocate_command_and_initialize<Render_Particles>(rq, kv.second._sim.get(), _renderParams);
	}
}

