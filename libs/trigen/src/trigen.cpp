// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "trigen.h"
#include "trigen.hpp"

#include <softbody.h>
#include <psp/psp.h>
#include <marching_cubes.h>

#include <list>
#include <memory>
#include <optional>

struct Trigen_Collider_t {
    sb::ISoftbody_Simulation::Collider_Handle handle;
};

struct Input_Texture {
    std::unique_ptr<uint8_t[]> data;
    PSP::Texture info;
};

struct Trigen_Session_t {
    sb::Unique_Ptr<sb::ISoftbody_Simulation> simulation;
    std::list<Trigen_Collider_t> colliders;

    std::vector<marching_cubes::metaball> _metaballs;

    marching_cubes::params _meshgenParams;
    float _metaballRadius;
    PSP::Parameters _paintParams;
    std::optional<PSP::Mesh> _pspMesh;

    PSP::Material _inputMaterial;
    Input_Texture _texBase;
    Input_Texture _texNormal;
    Input_Texture _texHeight;
    Input_Texture _texRoughness;
    Input_Texture _texAo;

    PSP::Material _outputMaterial;

    std::unique_ptr<PSP::IPainter> _painter;
};

extern "C" {

Trigen_Status TRIGEN_API Trigen_CreateSession(Trigen_Session *session, Trigen_Parameters const *params) {
    if (session == nullptr || params == nullptr) {
        return Trigen_InvalidArguments;
    }

    *session = nullptr;

    auto s = new Trigen_Session_t;
    if (s == nullptr) {
        return Trigen_OutOfMemory;
    }

    sb::Config sbConfig {};
    sb::Plant_Simulation_Extension_Extra sbPlant {};

    sbPlant.seed_position = glm::vec3(params->seed_position[0], params->seed_position[1], params->seed_position[2]);
    sbPlant.density = params->density;
    sbPlant.attachment_strength = params->attachment_strength;
    sbPlant.surface_adaption_strength = params->surface_adaption_strength;
    sbPlant.stiffness = params->stiffness;
    sbPlant.aging_rate = params->aging_rate;
    sbPlant.phototropism_response_strength = params->phototropism_response_strength;
    sbPlant.branching_probability = params->branching_probability;
    sbPlant.branch_angle_variance = params->branch_angle_variance;
    sbPlant.particle_count_limit = params->particle_count_limit;

    sbConfig.compute_preference = sb::Compute_Preference::GPU_Proprietary;
    if (params->flags & Trigen_F_PreferCPU) {
        sbConfig.compute_preference = sb::Compute_Preference::Reference;
    }

    sbConfig.ext = sb::Extension::Plant_Simulation;
    sbConfig.extra.plant_sim = &sbPlant;

    s->simulation = sb::create_simulation(sbConfig);
    if (s->simulation == nullptr) {
        delete s;
        return Trigen_InvalidConfiguration;
    }

    *session = s;
    return Trigen_OK;
}

Trigen_Status TRIGEN_API Trigen_DestroySession(Trigen_Session session) {
    if (session == nullptr) {
        return Trigen_InvalidArguments;
    }

    delete session;
    return Trigen_OK;
}

Trigen_Status TRIGEN_API Trigen_CreateCollider(Trigen_Collider *collider, Trigen_Session session, Trigen_Collider_Mesh const *mesh, Trigen_Transform const *transform) {
    return Trigen_OK;
}

Trigen_Status TRIGEN_API Trigen_UpdateCollider(Trigen_Collider collider, Trigen_Transform const *transform) {
    return Trigen_OK;
}

Trigen_Status TRIGEN_API Trigen_Grow(Trigen_Session session, float time) {
    return Trigen_OK;
}

Trigen_Status TRIGEN_API Trigen_Mesh_SetSubdivisions(Trigen_Session session, int subdivisions) {
    return Trigen_OK;
}

Trigen_Status TRIGEN_API Trigen_Mesh_Regenerate(Trigen_Session session) {
    return Trigen_OK;
}

Trigen_Status TRIGEN_API Trigen_Mesh_GetMesh(Trigen_Session session, Trigen_Mesh const *mesh) {
    return Trigen_OK;
}

Trigen_Status TRIGEN_API Trigen_Painting_SetInputTexture(Trigen_Session session, Trigen_Texture_Kind kind, Trigen_Texture const *texture) {
    return Trigen_OK;
}

Trigen_Status TRIGEN_API Trigen_Painting_SetOutputResolution(Trigen_Session session, int width, int height) {
    return Trigen_OK;
}

Trigen_Status TRIGEN_API Trigen_Painting_Regenerate(Trigen_Session session) {
    return Trigen_OK;
}

Trigen_Status TRIGEN_API Trigen_Painting_GetOutputTexture(Trigen_Session session, Trigen_Texture_Kind kind, Trigen_Texture const **texture) {
    return Trigen_OK;
}

}
