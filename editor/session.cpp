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

#include <imgui.h>
#include <ImGuizmo.h>

#include <trigen.h>

namespace fs = std::filesystem;

Session::Session(char const *name) :
    _name(name),
    _world(),
    _matView(1.0f),
    _matProj(1.0f) {
}

void Session::createPlant(Trigen_Parameters const &cfg) {
    _session.emplace(trigen::Session::make(cfg));
    printf("trigen::Session ptr is at %p\n", &_session.value());
    auto ent = _world.createEntity();
    _world.attachComponent<Plant_Component>(ent, &_session.value());
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

void Session::selectEntity(int index) {
    _selectedEntity = Entity_Handle(index);

    if (_world.getMapForComponent<Plant_Component>().count(_selectedEntity.value())) {
        emit meshgenAvailabilityChanged(true);
    } else {
        emit meshgenAvailabilityChanged(false);
    }
}

void Session::deselectEntity() {
    _selectedEntity.reset();
    emit meshgenAvailabilityChanged(false);
}

bool Session::selectedEntity(Entity_Handle *out) const {
    assert(out != nullptr);
    if (!_selectedEntity.has_value() || out == nullptr) {
        return false;
    }

    *out = *_selectedEntity;

    return true;
}

void Session::onTick(float deltaTime) {
    auto &transforms = _world.getMapForComponent<Transform_Component>();
    auto &colliders = _world.getMapForComponent<Collider_Component>();
    auto &plants = _world.getMapForComponent<Plant_Component>();
    auto &meshRenderables = _world.getMapForComponent<Mesh_Render_Component>();

    for (auto &colliderEnt : colliders) {
        // Check whether all colliders are added to all plant sims
        for (auto &plantEnt : plants) {
            if (colliderEnt.second.sb_handles.count(plantEnt.first) == 0) {
                auto collHandle = colliderEnt.second.mesh_collider->uploadToSimulation(*plantEnt.second.session);
                if (collHandle.has_value()) {
                    colliderEnt.second.sb_handles.emplace(std::make_pair(plantEnt.first, std::move(collHandle.value())));
                }
            }
        }

        // Collider has no mesh renderable yet; schedule a mesh upload
        if (meshRenderables.count(colliderEnt.first) == 0) {
            _pendingColliderMeshUploads.push_back(colliderEnt.first);
        }

        // Check if the collider's transform was manipulated
        if (transforms.count(colliderEnt.first) != 0) {
            auto &transform = transforms[colliderEnt.first];
            if (transform.manipulated) {
                Trigen_Transform tt;
                for (int i = 0; i < 3; i++) {
                    tt.position[i] = transform.position[i];
                    tt.scale[i] = transform.scale[i];
                }
                for (int i = 0; i < 4; i++) {
                    tt.orientation[i] = transform.rotation[i];
                }

                // If it was manipulated, we notify all the plants about this change
                for (auto &kv : colliderEnt.second.sb_handles) {
                    kv.second.update(tt);
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
            kv.second.session->grow(deltaTime);
        }
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

    if (_selectedEntity) {
        if (transforms.count(_selectedEntity.value())) {
            float mat[16];
            auto &transform = transforms[_selectedEntity.value()];

            ImGuizmo::RecomposeMatrixFromComponents(glm::value_ptr(transform.position), glm::value_ptr(transform.rotation), glm::value_ptr(transform.scale), mat);
            if (ImGuizmo::Manipulate(glm::value_ptr(_matView), glm::value_ptr(_matProj), gizmoOperation, ImGuizmo::MODE::WORLD, mat)) {
                ImGuizmo::DecomposeMatrixToComponents(mat, glm::value_ptr(transform.position), glm::value_ptr(transform.rotation), glm::value_ptr(transform.scale));
                transform.manipulated = true;
            }
        }
    }
}

void Session::onRender(topo::IRender_Queue *rq) {
    auto &transforms = _world.getMapForComponent<Transform_Component>();
    auto &plants = _world.getMapForComponent<Plant_Component>();
    auto &meshRenders = _world.getMapForComponent<Mesh_Render_Component>();

    // TODO: visualize branches
    // TODO: tint selected object

    // Find all entities that have both render info and a world transform
    for (auto &kv : meshRenders) {
        if (transforms.count(kv.first)) {
            auto &transform = transforms[kv.first];
            auto &meshRender = kv.second;
            topo::Transform topoTransform = {
                transform.position,
                transform.rotation,
                transform.scale
            };
            rq->Submit(meshRender.renderable, topoTransform);
        }
    }
}

void Session::setRunning(bool isRunning) {
    _isRunning = isRunning;
}

void Session::onMeshUpload(topo::IInstance *renderer) {
    if (_pendingColliderMeshUploads.size() > 0) {
        auto &colliders = _world.getMapForComponent<Collider_Component>();
        auto &meshRenderables = _world.getMapForComponent<Mesh_Render_Component>();

        for (auto handle : _pendingColliderMeshUploads) {
            assert(meshRenderables.count(handle) == 0);
            auto mdl = colliders[handle].mesh_collider->uploadToRenderer(renderer);
            topo::Material_ID material = nullptr;
            topo::Renderable_ID renderable = nullptr;
            renderer->CreateSolidColorMaterial(&material, { 1, 1, 1 });
            renderer->CreateRenderable(&renderable, mdl, material);
            meshRenderables[handle] = { renderable, mdl, material };
        }

        _pendingColliderMeshUploads.clear();
        _pendingColliderMeshUploads.shrink_to_fit();
    }
}

void Session::setGizmoMode(Session_Gizmo_Mode mode) {
    _gizmoMode = mode;
}

