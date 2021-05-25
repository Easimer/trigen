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
    float metaballScale;
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

TRIGEN_RETURN_CODE TRIGEN_API Trigen_CreateSession(
    TRIGEN_HANDLE_ACQUIRE Trigen_Session *session,
    TRIGEN_IN Trigen_Parameters const *params) {
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

    s->metaballScale = 8;
    s->_meshgenParams.subdivisions = 8;
    s->_paintParams.out_width = 512;
    s->_paintParams.out_height = 512;

    *session = s;
    return Trigen_OK;
}

TRIGEN_RETURN_CODE TRIGEN_API Trigen_DestroySession(
    TRIGEN_HANDLE_RELEASE Trigen_Session session) {
    assert(session);
    if (session == nullptr) {
        return Trigen_InvalidArguments;
    }

    delete session;
    return Trigen_OK;
}

TRIGEN_RETURN_CODE TRIGEN_API Trigen_CreateCollider(
    TRIGEN_HANDLE_ACQUIRE Trigen_Collider *collider,
    TRIGEN_HANDLE Trigen_Session session,
    TRIGEN_IN Trigen_Collider_Mesh const *mesh,
    TRIGEN_IN Trigen_Transform const *transform) {
    assert(collider && session && mesh && transform);
    if (collider == nullptr || session == nullptr || mesh == nullptr || transform == nullptr) {
        return Trigen_InvalidArguments;
    }

    *collider = nullptr;

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

TRIGEN_RETURN_CODE TRIGEN_API Trigen_UpdateCollider(
    TRIGEN_HANDLE Trigen_Collider collider,
    TRIGEN_IN Trigen_Transform const *transform) {
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

TRIGEN_RETURN_CODE TRIGEN_API Trigen_Grow(
    TRIGEN_HANDLE Trigen_Session session,
    tg_f32 time) {
    if (session == nullptr || time < 0 || std::isnan(time) || !std::isfinite(time)) {
        return Trigen_InvalidArguments;
    }

    session->simulation->step(time);

    return Trigen_OK;
}

TRIGEN_RETURN_CODE TRIGEN_API Trigen_Metaballs_SetScale(
    TRIGEN_HANDLE Trigen_Session session,
    tg_f32 scale) {
    assert(session);
    assert(radius > 0.0f);

    if (session == nullptr || scale < 0.0f) {
        return Trigen_InvalidArguments;
    }

    session->metaballScale = scale;

    return Trigen_OK;
}

TRIGEN_RETURN_CODE TRIGEN_API Trigen_Metaballs_Regenerate(
    TRIGEN_HANDLE Trigen_Session session) {
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
            float radius = 0;
            for (int i = 0; i < 3; i++) {
                radius = glm::max(size[i] / 2, radius);
            }
            session->_metaballs.push_back({ p, radius / 8, session->metaballScale });
        }
    }

    return Trigen_OK;
}

TRIGEN_RETURN_CODE TRIGEN_API Trigen_Mesh_SetSubdivisions(
    TRIGEN_HANDLE Trigen_Session session,
    tg_u32 subdivisions) {
    assert(session);
    assert(subdivisions >= 1);
    if (session == nullptr || subdivisions < 1) {
        return Trigen_InvalidArguments;
    }

    session->_meshgenParams.subdivisions = subdivisions;
    return Trigen_OK;
}

TRIGEN_RETURN_CODE TRIGEN_API Trigen_Mesh_Regenerate(
    TRIGEN_HANDLE Trigen_Session session) {
    assert(session);
    if (session == nullptr) {
        return Trigen_InvalidArguments;
    }

    if (session->_metaballs.size() == 0) {
        return Trigen_NotReady;
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

TRIGEN_RETURN_CODE TRIGEN_API Trigen_Mesh_GetMesh(
    TRIGEN_HANDLE Trigen_Session session,
    TRIGEN_IN Trigen_Mesh *outMesh) {
    assert(session && outMesh);
    if (session == nullptr || outMesh == nullptr) {
        return Trigen_InvalidArguments;
    }

    if (!session->_pspMesh.has_value()) {
        return Trigen_NotReady;
    }

    auto &mesh = session->_pspMesh.value();

    outMesh->position_count = mesh.position.size();
    outMesh->normal_count = mesh.normal.size();
    outMesh->triangle_count = mesh.elements.size() / 3;

    outMesh->positions = new float[mesh.position.size() * 3];
    outMesh->normals = new float[mesh.normal.size() * 3];
    outMesh->uvs = new float[mesh.uv.size() * 2];
    outMesh->vertex_indices = new uint64_t[mesh.elements.size() * 3];
    // PSP::Mesh has no normal indices
    outMesh->normal_indices = outMesh->vertex_indices;

    memcpy((void*)outMesh->positions, mesh.position.data(), outMesh->position_count * sizeof(glm::vec3));
    memcpy((void*)outMesh->normals, mesh.normal.data(), outMesh->normal_count * sizeof(glm::vec3));
    memcpy((void*)outMesh->uvs, mesh.uv.data(), mesh.uv.size() * sizeof(glm::vec2));

    static_assert(sizeof(uint64_t) == sizeof(size_t), "size_t was not 8 bytes! Fix this memcpy:");
    memcpy((void *)outMesh->vertex_indices, mesh.elements.data(), mesh.elements.size() * sizeof(uint64_t));

    return Trigen_OK;
}

TRIGEN_RETURN_CODE TRIGEN_API Trigen_Mesh_FreeMesh(
    TRIGEN_FREED TRIGEN_INOUT Trigen_Mesh *mesh) {
    if (mesh != nullptr) {
        delete[] mesh->vertex_indices;
        // PSP::Mesh has no normal indices
        // delete[] mesh->normal_indices;
        delete[] mesh->positions;
        delete[] mesh->normals;
        delete[] mesh->uvs;

        memset(mesh, 0, sizeof(*mesh));
    }

    return Trigen_OK;
}

TRIGEN_RETURN_CODE TRIGEN_API Trigen_Painting_SetInputTexture(
    TRIGEN_HANDLE Trigen_Session session,
    Trigen_Texture_Kind kind,
    TRIGEN_IN Trigen_Texture const *texture) {
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

TRIGEN_RETURN_CODE TRIGEN_API Trigen_Painting_SetOutputResolution(
    TRIGEN_HANDLE Trigen_Session session,
    tg_u32 width,
    tg_u32 height) {
    assert(session);
    if (session == nullptr || width == 0 || height == 0) {
        return Trigen_InvalidArguments;
    }

    session->_paintParams.out_width = width;
    session->_paintParams.out_height = height;

    return Trigen_OK;
}

TRIGEN_RETURN_CODE TRIGEN_API Trigen_Painting_Regenerate(
    TRIGEN_HANDLE Trigen_Session session) {
    assert(session);
    if (session == nullptr) {
        return Trigen_InvalidArguments;
    }

    if (!session->_pspMesh.has_value()) {
        return Trigen_NotReady;
    }

    PSP::Texture texBlack = {};
    uint8_t const blackPixel[3] = { 0, 0, 0 };

    texBlack.buffer = blackPixel;
    texBlack.width = 1;
    texBlack.height = 1;

    PSP::Texture texNormal = {};
    uint8_t const normalPixel[3] = { 128, 128, 255 };

    texNormal.buffer = normalPixel;
    texNormal.width = 1;
    texNormal.height = 1;

    auto putPlaceholderTextureIfEmpty = [&](Input_Texture const &input, PSP::Texture &target, PSP::Texture const &placeholder) {
        if (input.data == nullptr) {
            target = placeholder;
        } else {
            target = input.info;
        }
    };

    auto putBlackTextureIfEmpty = std::bind(putPlaceholderTextureIfEmpty, std::placeholders::_1, std::placeholders::_2, texBlack);
    auto putNormalTextureIfEmpty = std::bind(putPlaceholderTextureIfEmpty, std::placeholders::_1, std::placeholders::_2, texNormal);

    assert(session->_pspMesh->uv.size() == session->_pspMesh->elements.size());

    putBlackTextureIfEmpty(session->_texBase, session->_inputMaterial.base);
    putNormalTextureIfEmpty(session->_texNormal, session->_inputMaterial.normal);
    putBlackTextureIfEmpty(session->_texHeight, session->_inputMaterial.height);
    putBlackTextureIfEmpty(session->_texRoughness, session->_inputMaterial.roughness);
    putBlackTextureIfEmpty(session->_texAo, session->_inputMaterial.ao);

    session->_paintParams.material = &session->_inputMaterial;
    session->_paintParams.mesh = &session->_pspMesh.value();

    session->_outputMaterial = {};
    session->_painter = PSP::make_painter(session->_paintParams);

    session->_painter->step_painting(0);

    if (session->_painter->is_painting_done()) {
        session->_painter->result(&session->_outputMaterial);
    }

    return Trigen_OK;
}

TRIGEN_RETURN_CODE TRIGEN_API Trigen_Painting_GetOutputTexture(
    TRIGEN_HANDLE Trigen_Session session,
    Trigen_Texture_Kind kind,
    TRIGEN_INOUT Trigen_Texture *texture) {
    assert(session);
    assert(Trigen_Texture_BaseColor <= kind && kind < Trigen_Texture_Max);
    assert(texture);
    if (session == nullptr ||
        !(Trigen_Texture_BaseColor <= kind && kind < Trigen_Texture_Max) ||
        texture == nullptr) {
        return Trigen_InvalidArguments;
    }

    PSP::Texture *tex = nullptr;
    switch (kind) {
    case Trigen_Texture_BaseColor:
        tex = &session->_outputMaterial.base;
        break;
    case Trigen_Texture_NormalMap:
        tex = &session->_outputMaterial.normal;
        break;
    case Trigen_Texture_HeightMap:
        tex = &session->_outputMaterial.height;
        break;
    case Trigen_Texture_RoughnessMap:
        tex = &session->_outputMaterial.roughness;
        break;
    case Trigen_Texture_AmbientOcclusionMap:
        tex = &session->_outputMaterial.ao;
        break;
    default:
        assert(0);
        return Trigen_InvalidArguments;
    }

    if (tex->buffer == nullptr) {
        return Trigen_NotReady;
    }

    texture->width = tex->width;
    texture->height = tex->height;
    texture->image = tex->buffer;

    return Trigen_OK;
}

TRIGEN_RETURN_CODE TRIGEN_API Trigen_GetErrorMessage(
    TRIGEN_OUT char const **message,
    Trigen_Status rc) {
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

TRIGEN_RETURN_CODE TRIGEN_API Trigen_GetBranches(
    TRIGEN_HANDLE Trigen_Session session,
    TRIGEN_INOUT tg_usize *count,
    TRIGEN_IN tg_f32 *buffer) {
    if (session == nullptr || count == nullptr) {
        return Trigen_InvalidArguments;
    }

    size_t numBranches = 0;
    auto iter = session->simulation->get_connections();

    // Count the number of branches
    while (!iter->ended()) {
        numBranches++;
        iter->step();
    }

    if (*count >= numBranches && buffer != nullptr) {
        // Buffer's pointer is not null and it has enough space

        iter = session->simulation->get_connections();

        // Copy branch endpoint positions
        while (!iter->ended()) {
            auto c = iter->get();
            for (int i = 0; i < 3; i++) {
                buffer[0 + i] = c.parent_position[i];
                buffer[3 + i] = c.child_position[i];
            }

            buffer += 6;
            iter->step();
        }

        return Trigen_OK;
    } else {
        if (*count == 0) {
            *count = numBranches;
            return Trigen_OK;
        } else {
            if (buffer == nullptr) {
                // Count is non-zero, but buffer is null
                return Trigen_InvalidArguments;
            } else {
                // Buffer is non-null, but count is less than numBranches
                return Trigen_NotEnoughSpace;
            }
        }
    }

    return Trigen_OK;
}

}

