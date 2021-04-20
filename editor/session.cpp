// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"
#include "session.h"
#include "world.h"
#include "collimp.h"
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>
#include <unordered_set>
#include <filesystem>
#include <variant>

#include <r_cmd/general.h>
#include <r_cmd/softbody.h>

#include <imgui.h>
#include <ImGuizmo.h>

namespace fs = std::filesystem;

Session::Session(char const *name) :
	_name(name),
	_renderParams{},
	_world(),
    _matView(1.0f),
    _matProj(1.0f) {
}

void Session::createPlant(sb::Config const &cfg) {
	auto ent = _world.createEntity();
	_world.attachComponent<Plant_Component>(ent, cfg);
}

void Session::addColliderFromPath(char const *path) {
	fs::path P(path);
	auto ext = P.extension();
	Mesh_Source_Kind meshSourceKind;
	if (ext == ".obj") {
		meshSourceKind = Mesh_Source_Kind::OBJ;
	} else {
		fprintf(stderr, "Can't determine the model type of '%s'\n", path);
		return;
	}

	auto importer = makeColliderImporter(meshSourceKind);
	auto colliders = importer->loadFromFile(path);

	for (auto &collider : colliders) {
		auto entityHandle = _world.createEntity();

		auto componentTransform = _world.attachComponent<Transform_Component>(entityHandle);
		componentTransform->position = collider->transform().position;
		componentTransform->rotation = collider->transform().rotation;
		componentTransform->scale = collider->transform().scale;

		auto componentCollider = _world.attachComponent<Collider_Component>(entityHandle);
		componentCollider->mesh_collider = std::move(collider);
	}
}

void Session::onTick(float deltaTime) {
	auto &transforms = _world.getMapForComponent<Transform_Component>();
	auto &colliders = _world.getMapForComponent<Collider_Component>();
	auto &plants = _world.getMapForComponent<Plant_Component>();
	auto &untexturedMeshRenderables = _world.getMapForComponent<Untextured_Mesh_Render_Component>();

	for (auto &colliderEnt : colliders) {
        // Check whether all colliders are added to all plant sims
		for (auto &plantEnt : plants) {
			if (colliderEnt.second.sb_handles.count(plantEnt.first) == 0) {
				auto collHandle = colliderEnt.second.mesh_collider->uploadToSimulation(plantEnt.second._sim.get());
                colliderEnt.second.sb_handles.emplace(std::make_pair(plantEnt.first, collHandle));
			}
		}

		if (untexturedMeshRenderables.count(colliderEnt.first) == 0) {
			_pendingColliderMeshUploads.push_back(colliderEnt.first);
		}

		if (transforms.count(colliderEnt.first) != 0) {
			auto &transform = transforms[colliderEnt.first];
			if (transform.manipulated) {
				for (auto &plantEnt : plants) {
					auto handle = colliderEnt.second.sb_handles[plantEnt.first];
					auto matTransform = glm::translate(transform.position) * glm::mat4(transform.rotation) * glm::scale(transform.scale);
					plantEnt.second._sim->update_transform(handle, matTransform);
				}
			}
			transform.manipulated = false;
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
		if (_isRunning) {
			kv.second._sim->step(deltaTime);
		}
	}
}

void Session::onRender(gfx::Render_Queue *rq) {
	auto &transforms = _world.getMapForComponent<Transform_Component>();
	auto &plants = _world.getMapForComponent<Plant_Component>();
	auto &meshRenders = _world.getMapForComponent<Mesh_Render_Component>();
	auto &untexturedMeshRenders = _world.getMapForComponent<Untextured_Mesh_Render_Component>();

	gfx::allocate_command_and_initialize<Fetch_Camera_Matrices>(rq, &_matView, &_matProj);

	gfx::allocate_command_and_initialize<Render_Grid>(rq);

	for (auto &kv : plants) {
		gfx::allocate_command_and_initialize<Visualize_Connections>(rq, kv.second._sim.get());
	}

	using Mesh_Render = std::variant<Mesh_Render_Component *, Untextured_Mesh_Render_Component *>;

	// Find all entities that have both render info and a world transform
	std::vector<std::pair<Mesh_Render, Transform_Component *>> renderables;
	for (auto &kv : untexturedMeshRenders) {
		if (transforms.count(kv.first)) {
			renderables.emplace_back(std::make_pair(&kv.second, &transforms.at(kv.first)));
		}
	}

	for (auto &kv : meshRenders) {
		if (transforms.count(kv.first)) {
			renderables.emplace_back(std::make_pair(&kv.second, &transforms.at(kv.first)));
		}
	}

	for (auto &renderable : renderables) {
		std::visit([&](auto &&arg) {
			using T = std::decay_t<decltype(arg)>;
            auto mdl = arg->model;
            auto transform = gfx::Transform{
                renderable.second->position,
                renderable.second->rotation,
                renderable.second->scale
            };
			if constexpr (std::is_same_v<T, Mesh_Render_Component *>) {
				auto texDiffuse = arg->material.diffuse;
                gfx::allocate_command_and_initialize<Render_Model>(rq, mdl, texDiffuse, transform);
			} else if constexpr (std::is_same_v<T, Untextured_Mesh_Render_Component *>) {
                gfx::allocate_command_and_initialize<Render_Untextured_Model>(rq, mdl, transform);
			}
        }, renderable.first);
	}

	ImGuizmo::OPERATION gizmoOperation = ImGuizmo::OPERATION::TRANSLATE;

	switch (_gizmoMode) {
	case Session_Gizmo_Mode::Translation:
		gizmoOperation = ImGuizmo::OPERATION::TRANSLATE;
		break;
	case Session_Gizmo_Mode::Rotation:
		gizmoOperation = ImGuizmo::OPERATION::ROTATE;
		break;
	case Session_Gizmo_Mode::Scaling:
		gizmoOperation = ImGuizmo::OPERATION::SCALE;
		break;
	}

    for (auto &transform : transforms) {
        float mat[16];
        ImGuizmo::RecomposeMatrixFromComponents(glm::value_ptr(transform.second.position), glm::value_ptr(transform.second.rotation), glm::value_ptr(transform.second.scale), mat);
        if (ImGuizmo::Manipulate(glm::value_ptr(_matView), glm::value_ptr(_matProj), gizmoOperation, ImGuizmo::MODE::WORLD, mat)) {
            ImGuizmo::DecomposeMatrixToComponents(mat, glm::value_ptr(transform.second.position), glm::value_ptr(transform.second.rotation), glm::value_ptr(transform.second.scale));
            transform.second.manipulated = true;
        }
    }
}

void Session::setRunning(bool isRunning) {
	_isRunning = isRunning;
}

class Collider_Mesh_Upload : public gfx::IRender_Command {
public:
	Collider_Mesh_Upload(Entity_Handle handle, Collider_Component *collider, std::unordered_map<Entity_Handle, Untextured_Mesh_Render_Component> *meshRenderables) :
	_handle(handle), _collider(collider), _meshRenderables(meshRenderables) {
	}

	void execute(gfx::IRenderer *renderer) override {
		auto mdl = _collider->mesh_collider->uploadToRenderer(renderer);
		(*_meshRenderables)[_handle] = Untextured_Mesh_Render_Component{ mdl };
	}
private:
	Entity_Handle _handle;
	Collider_Component *_collider;
	std::unordered_map<Entity_Handle, Untextured_Mesh_Render_Component> *_meshRenderables;
};

void Session::onMeshUpload(gfx::Render_Queue *rq) {
	if (_pendingColliderMeshUploads.size() > 0) {
        auto &colliders = _world.getMapForComponent<Collider_Component>();
        auto &meshRenderables = _world.getMapForComponent<Untextured_Mesh_Render_Component>();

		for (auto handle : _pendingColliderMeshUploads) {
			assert(meshRenderables.count(handle) == 0);
			gfx::allocate_command_and_initialize<Collider_Mesh_Upload>(rq, handle, &colliders[handle], &meshRenderables);
		}

		_pendingColliderMeshUploads.clear();
		_pendingColliderMeshUploads.shrink_to_fit();
	}
}

void Session::setGizmoMode(Session_Gizmo_Mode mode) {
	_gizmoMode = mode;
}

