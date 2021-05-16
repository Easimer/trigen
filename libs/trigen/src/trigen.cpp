// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "trigen.h"
#include "trigen.hpp"

#include <marching_cubes.h>
#include <psp/psp.h>
#include <softbody.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <list>
#include <memory>
#include <optional>
#include <set>
#include <vector>

#include <glm/gtc/type_ptr.hpp>

struct Trigen_Collider_t {
    sb::ISoftbody_Simulation *simulation;
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
    float metaballRadius;
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
    assert(session && params);
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
    assert(session);
    if (session == nullptr) {
        return Trigen_InvalidArguments;
    }

    delete session;
    return Trigen_OK;
}

Trigen_Status TRIGEN_API Trigen_CreateCollider(Trigen_Collider *collider, Trigen_Session session, Trigen_Collider_Mesh const *mesh, Trigen_Transform const *transform) {
    assert(collider && session && mesh && transform);
    if (collider == nullptr || session == nullptr || mesh == nullptr || transform == nullptr) {
        return Trigen_InvalidArguments;
    }

    assert(mesh->normals && mesh->positions && mesh->vertex_indices && mesh->normal_indices);
    if (mesh->normals == nullptr || mesh->positions == nullptr || mesh->vertex_indices == nullptr || mesh->normal_indices == nullptr) {
        return Trigen_InvalidMesh;
    }

    sb::Mesh_Collider sbCollider;
    sbCollider.triangle_count = mesh->triangle_count;
    sbCollider.vertex_indices = mesh->vertex_indices;
    sbCollider.normal_indices = mesh->normal_indices;
    sbCollider.position_count = mesh->position_count;
    sbCollider.positions = mesh->positions;
    sbCollider.normal_count = mesh->normal_count;
    sbCollider.normals = mesh->normals;

    auto matOrient = glm::mat4_cast(glm::quat(transform->orientation[0], transform->orientation[1], transform->orientation[2], transform->orientation[3]));
    auto matTransform = glm::translate(matOrient, glm::vec3(transform->position[0], transform->position[1], transform->position[2]));

    sbCollider.transform = matTransform;

    sb::ISoftbody_Simulation::Collider_Handle handle;
    if (!session->simulation->add_collider(handle, &sbCollider)) {
        return Trigen_Failure;
    }

    session->colliders.push_front({ session->simulation.get(), handle });

    return Trigen_OK;
}

Trigen_Status TRIGEN_API Trigen_UpdateCollider(Trigen_Collider collider, Trigen_Transform const *transform) {
    assert(collider && transform);
    if (collider == nullptr || transform == nullptr) {
        return Trigen_InvalidArguments;
    }

    assert(collider->simulation);

    if (collider->simulation == nullptr) {
        return Trigen_InvalidArguments;
    }

    auto matOrient = glm::mat4_cast(glm::quat(transform->orientation[0], transform->orientation[1], transform->orientation[2], transform->orientation[3]));
    auto matTransform = glm::translate(matOrient, glm::vec3(transform->position[0], transform->position[1], transform->position[2]));

    if (!collider->simulation->update_transform(collider->handle, matTransform)) {
        return Trigen_Failure;
    }

    return Trigen_OK;
}

Trigen_Status TRIGEN_API Trigen_Grow(Trigen_Session session, float time) {
    if (session == nullptr || time < 0 || isnan(time) || !isfinite(time)) {
        return Trigen_InvalidArguments;
    }

    session->simulation->step(time);

    return Trigen_OK;
}

Trigen_Status TRIGEN_API Trigen_Metaballs_SetRadius(Trigen_Session session, float radius) {
    assert(session);
    assert(radius > 0.0f);

    if (session == nullptr || radius < 0.0f) {
        return Trigen_InvalidArguments;
    }

    session->metaballRadius = radius;

    return Trigen_OK;
}

Trigen_Status TRIGEN_API Trigen_Metaballs_Regenerate(Trigen_Session session) {
    assert(session);

    if (session == nullptr) {
        return Trigen_InvalidArguments;
    }

    session->_metaballs.clear();

    auto &simulation = session->simulation;

    // Gather particles and connections
    std::unordered_map<sb::index_t, sb::Particle> particles;
    std::set<std::pair<sb::index_t, sb::index_t>> connections;

    for (auto iter = simulation->get_particles(); !iter->ended(); iter->step()) {
        auto p = iter->get();
        particles[p.id] = p;
    }

    for (auto iter = simulation->get_connections(); !iter->ended(); iter->step()) {
        auto c = iter->get();
        if (c.parent < c.child) {
            connections.insert({ c.parent, c.child });
        } else {
            connections.insert({ c.child, c.parent });
        }
    }

    // Generate metaballs
    for (auto &conn : connections) {
        auto p0 = particles[conn.first].position;
        auto p1 = particles[conn.second].position;
        auto s0 = particles[conn.first].size;
        auto s1 = particles[conn.second].size;
        auto dir = p1 - p0;
        auto dirLen = length(p1 - p0);
        auto sizDir = s1 - s0;
        auto steps = int((dirLen + 1) * 16.0f);

        for (int s = 0; s < steps; s++) {
            auto t = s / (float)steps;
            auto p = p0 + t * dir;
            auto size = s0 + t * sizDir;
            float radius = session->metaballRadius;
            for (int i = 0; i < 3; i++) {
                radius = glm::max(size[i] / 2, radius);
            }
            session->_metaballs.push_back({ p, radius / 8 });
        }
    }

    return Trigen_OK;
}

Trigen_Status TRIGEN_API Trigen_Mesh_SetSubdivisions(Trigen_Session session, int subdivisions) {
    assert(session);
    assert(subdivisions >= 1);
    if (session == nullptr || subdivisions < 1) {
        return Trigen_InvalidArguments;
    }

    session->_meshgenParams.subdivisions = subdivisions;
    return Trigen_OK;
}

Trigen_Status TRIGEN_API Trigen_Mesh_Regenerate(Trigen_Session session) {
    assert(session);
    if (session == nullptr) {
        return Trigen_InvalidArguments;
    }

    auto mesh = marching_cubes::generate(session->_metaballs, session->_meshgenParams);

    PSP::Mesh pspMesh;
    pspMesh.position = std::move(mesh.positions);
    pspMesh.normal = std::move(mesh.normal);
    std::transform(mesh.indices.begin(), mesh.indices.end(), std::back_inserter(pspMesh.elements), [&](unsigned idx) { return (size_t)idx; });

    session->_pspMesh.emplace(std::move(pspMesh));

    // Regenerate UVs
    session->_pspMesh->uv.clear();
    PSP::unwrap(session->_pspMesh.value());

    return Trigen_OK;
}

Trigen_Status TRIGEN_API Trigen_Mesh_GetMesh(Trigen_Session session, Trigen_Mesh *mesh) {
    assert(session && mesh);
    if (session == nullptr || mesh == nullptr) {
        return Trigen_InvalidArguments;
    }

    if (!session->_pspMesh.has_value()) {
        return Trigen_NotReady;
    }

    mesh->position = (float*)session->_pspMesh->position.data();

    return Trigen_OK;
}

Trigen_Status TRIGEN_API Trigen_Painting_SetInputTexture(Trigen_Session session, Trigen_Texture_Kind kind, Trigen_Texture const *texture) {
    assert(session);
    assert(Trigen_Texture_BaseColor <= kind && kind < Trigen_Texture_Max);
    assert(texture);
    assert(texture->image);
    if (session == nullptr ||
        !(Trigen_Texture_BaseColor <= kind && kind < Trigen_Texture_Max) ||
        texture == nullptr ||
        texture->image == nullptr) {
        return Trigen_InvalidArguments;
    }

    // TODO: assuming RGB888 here with no padding
    auto size = texture->width * texture->height * 3;

    Input_Texture *tex = nullptr;
    switch (kind) {
    case Trigen_Texture_BaseColor:
        tex = &session->_texBase;
        break;
    case Trigen_Texture_NormalMap:
        tex = &session->_texNormal;
        break;
    case Trigen_Texture_HeightMap:
        tex = &session->_texHeight;
        break;
    case Trigen_Texture_RoughnessMap:
        tex = &session->_texRoughness;
        break;
    case Trigen_Texture_AmbientOcclusionMap:
        tex = &session->_texAo;
        break;
    default:
        assert(0);
        break;
    }

    if (tex != nullptr) {
        auto data = std::make_unique<uint8_t[]>(size);
        memcpy(data.get(), texture->image, size);
        tex->data = std::move(data);
        tex->info.buffer = tex->data.get();
        tex->info.width = texture->width;
        tex->info.height = texture->height;
    }

    return Trigen_OK;
}

Trigen_Status TRIGEN_API Trigen_Painting_SetOutputResolution(Trigen_Session session, int width, int height) {
    assert(session);
    if (session == nullptr) {
        return Trigen_InvalidArguments;
    }

    session->_paintParams.out_width = width;
    session->_paintParams.out_height = height;

    return Trigen_OK;
}

Trigen_Status TRIGEN_API Trigen_Painting_Regenerate(Trigen_Session session) {
    assert(session);
    if (session == nullptr) {
        return Trigen_InvalidArguments;
    }

    return Trigen_OK;
}

Trigen_Status TRIGEN_API Trigen_Painting_GetOutputTexture(Trigen_Session session, Trigen_Texture_Kind kind, Trigen_Texture const **texture) {
    assert(session);
    return Trigen_OK;
}

Trigen_Status TRIGEN_API Trigen_GetErrorMessage(char const **message, Trigen_Status rc) {
    if (message == nullptr) {
        return Trigen_InvalidArguments;
    }

    switch (rc) {
    case Trigen_OK:
        *message = "No error";
        break;
    case Trigen_Failure:
        *message = "General failure";
        break;
    case Trigen_InvalidArguments:
        *message = "One or more arguments have an invalid value";
        break;
    case Trigen_OutOfMemory:
        *message = "Out of memory";
        break;
    case Trigen_InvalidConfiguration:
        *message = "One or more parameters in the session configuration struct have an invalid value";
        break;
    case Trigen_InvalidMesh:
        *message = "One or more parameters in the mesh descriptor have an invalid value";
        break;
    case Trigen_NotReady:
        *message = "Can't run this generation phase because a previous one hasn't been run yet";
        break;
    default:
        *message = "Unknown error code";
        return Trigen_InvalidArguments;
    }

    return Trigen_OK;
}

}

