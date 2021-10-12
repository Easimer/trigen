// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#include "scene_loader.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <stb_image.h>

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
LoadSceneFromFile(
    std::string const &path,
    Scene &scene,
    topo::IInstance *renderer,
    Trigen_Session *session,
    std::vector<Scene::Collider> &colliders,
    Demo &demo) {
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

    auto elemDemo = J["demo"];
    auto elemKind = elemDemo["kind"];

    if (elemKind == "oneshot") {
        demo.kind = Demo::ONESHOT;
        demo.oneshot.at = ValueOrDefault(elemDemo, "at", 0.0f);
        demo.oneshot.hold = ValueOrDefault(elemDemo, "hold", 1.0f);
    } else if (elemKind == "timelapse") {
        demo.kind = Demo::TIMELAPSE;
        demo.timelapse.from = ValueOrDefault(elemDemo, "from", 0.0f);
        demo.timelapse.to = ValueOrDefault(elemDemo, "to", 10.0f);
        demo.timelapse.step = ValueOrDefault(elemDemo, "step", 1.0f);
        demo.timelapse.stepFrequency
            = ValueOrDefault(elemDemo, "stepFrequency", 2.0f);
    } else {
        demo.kind = Demo::NONE;
    }

    auto elemPainting = J["painting"];
    auto elemDiffuse = elemPainting["diffuse"];
    auto elemNormal = elemPainting["normal"];
    if (!elemDiffuse.is_string() || !elemNormal.is_string()) {
        throw Scene_Loader_Exception(
            "One or more painting parameters are missing!");
    }

    auto loadTexture = [session](
                           std::string const &path,
                           Trigen_Texture_Kind kind) {
        int width, height, ch;
        auto image = stbi_load(path.c_str(), &width, &height, &ch, 3);

        if (image == nullptr) {
            throw Scene_Loader_Exception("Couldn't load " + path + "!");
        }

        Trigen_Texture texture;
        texture.width = width;
        texture.height = height;
        texture.image = image;
        Trigen_Painting_SetInputTexture(
            *session, kind, &texture);

        stbi_image_free(image);
    };


    loadTexture(elemDiffuse, Trigen_Texture_BaseColor);
    loadTexture(elemNormal, Trigen_Texture_NormalMap);

    return true;
}
