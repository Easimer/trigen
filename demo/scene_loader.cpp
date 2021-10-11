// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#include "scene_loader.h"

#include <nlohmann/json.hpp>
#include <fstream>

static void
MakeColliderObject(
    std::vector<Scene::Collider> &colliderAccumulator,
    nlohmann::json const &J,
    Scene &scene,
    topo::IInstance *renderer,
    Trigen_Session session) {
    Trigen_Transform transform;
    for (int i = 0; i < 3; i++) {
        transform.position[i] = 0;
        transform.scale[i] = 1;
    }
    for (int i = 0; i < 4; i++) {
        transform.orientation[i] = 0;
    }
    transform.orientation[0] = 1;

    auto const &position = J["position"];
    auto const &orientation = J["orientation"];
    auto const &scale = J["scale"];
    if (position.is_array() && position.size() >= 3) {
        for (int i = 0; i < 3; i++) {
            transform.position[i] = (float)position[i];
        }
    }
    if (orientation.is_array() && orientation.size() >= 4) {
        for (int i = 0; i < 4; i++) {
            transform.orientation[i] = (float)orientation[i];
        }
    }
    if (scale.is_array() && scale.size() >= 4) {
        for (int i = 0; i < 3; i++) {
            transform.scale[i] = (float)scale[i];
        }
    }

    auto const &file = J["file"];
    if (!file.is_string()) {
        throw Scene_Loader_Exception(
            "Path to environment object mesh is not a string!");
    }
    auto path = (std::string)file;

    auto colliders = scene.LoadObjMeshCollider(renderer, session, transform, path.c_str());
    while (!colliders.empty()) {
        colliderAccumulator.emplace_back(std::move(colliders.back()));
        colliders.pop_back();
    }
}

template<typename T>
static T
ValueOrDefault(nlohmann::json const &J, const char *key, T value) {
    if (J[key].is_number()) {
        return (T)J[key];
    } else {
        return value;
    }
}

bool
LoadSceneFromFile(std::string const &path, Scene &scene, topo::IInstance *renderer, Trigen_Session *session, std::vector<Scene::Collider> &colliders) {
    std::ifstream stream(path);
    if (!stream) {
        return false;
    }

    nlohmann::json J;
    stream >> J;

    if (J["simulation"].is_null()) {
        fprintf(
            stderr, "Can't load scene from '%s': key 'simulation' is missing\n",
            path.c_str());
        return false;
    }

    if (J["environment"].is_null()) {
        fprintf(
            stderr,
            "Can't load scene from '%s': key 'environment' is missing\n",
            path.c_str());
        return false;
    }

    auto const &simulation = J["simulation"];

    Trigen_Parameters params = {};
    params.flags = 0;

    if (simulation["preferCPU"].is_boolean() && (bool)simulation["preferCPU"]) {
        params.flags |= Trigen_F_PreferCPU;
    }

    auto const &seed_position = simulation["seed_position"];
    if (seed_position.is_array() && seed_position.size() >= 3) {
        for (int i = 0; i < 3; i++)
            params.seed_position[i] = (float)seed_position[i];
    } else {
        for (int i = 0; i < 3; i++)
            params.seed_position[i] = 0;
    }

    params.particle_count_limit
        = ValueOrDefault(simulation, "particleCountLimit", tg_u32(512));
    params.density = ValueOrDefault(simulation, "density", 1.0f);
    params.phototropism_response_strength
        = ValueOrDefault(simulation, "phototropismResponseStrength", 1.0f);
    params.aging_rate = ValueOrDefault(simulation, "agingRate", 0.1f);
    params.surface_adaption_strength
        = ValueOrDefault(simulation, "surfaceAdaptionStrength", 1.0f);
    params.attachment_strength
        = ValueOrDefault(simulation, "attachmentStrength", 1.0f);
    params.stiffness = ValueOrDefault(simulation, "stiffness", 0.2f);

    auto branching = simulation["branching"];
    if (branching.is_object()) {
        params.branching_probability
            = ValueOrDefault(branching, "probability", 0.5f);
        params.branch_angle_variance
            = ValueOrDefault(branching, "angleVariance", 3.1415926f);
    } else {
        params.branching_probability = 0.5f;
        params.branch_angle_variance = 3.1415926f;
    }

    Trigen_Status rc;
    if ((rc = Trigen_CreateSession(session, &params)) != Trigen_OK) {
        throw Scene_Loader_Exception(rc);
    }

    auto env = J["environment"];
    for (auto x : env) {
        if (!x.is_object()) {
            throw Scene_Loader_Exception("Element of environment array is not an object!");
        }
        auto kind = x["kind"];
        if (!kind.is_string()) {
            throw Scene_Loader_Exception("Environment object kind is not a string!");
        }

        if (kind == "collider") {
            MakeColliderObject(colliders, x, scene, renderer, *session);
        } else {
            throw Scene_Loader_Exception(
                "Unknown environment object kind '" + (std::string)kind + "'");
        }
    }

    return true;
}
