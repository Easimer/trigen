// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "trigen.h"
#include "trigen.hpp"

#include <foliage.hpp>
#include <marching_cubes.h>
#include <psp/psp.h>
#include <softbody.h>
#include <trigen/mesh_compress.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <list>
#include <memory>
#include <optional>
#include <set>
#include <unordered_set>
#include <vector>

#include <glm/gtc/type_ptr.hpp>

struct Trigen_Collider_t {
    sb::ISoftbody_Simulation *simulation;
    sb::ISoftbody_Simulation::Collider_Handle handle;
};

/** \brief An input texture; owns both the pixel data and the
 * PSP::Input_Texture descriptor.
 */
struct Input_Texture {
    std::unique_ptr<uint8_t[]> data;
    PSP::Input_Texture info;
};

struct Trigen_Session_t {
    sb::Unique_Ptr<sb::ISoftbody_Simulation> simulation;
    std::list<Trigen_Collider_t> colliders;

    std::vector<marching_cubes::metaball> _metaballs;

    marching_cubes::params _meshgenParams;
    float metaballScale;

    std::optional<PSP::Mesh> _pspMesh;

    std::vector<Trigen_Texture_Slot_Descriptor> _texSlots;
    bool _texSlotsLocked = false;

    std::vector<Input_Texture> _inputTextures;

    tg_u32 _outputWidth;
    tg_u32 _outputHeight;
    PSP::Output_Material _outputMaterial;

    std::unique_ptr<IFoliage_Generator> foliageGenerator;
    std::vector<tg_u64> foliageIndexBuffer;
};

#define TEXSLOTDESC_INIT_RGB888(slot, r, g, b) \
    slot.defaultPixel.rgb888[0] = r; \
    slot.defaultPixel.rgb888[1] = g; \
    slot.defaultPixel.rgb888[2] = b;

#define TEXSLOTDESC_INIT_RGB888_BLACK(slot) TEXSLOTDESC_INIT_RGB888(slot, 0, 0, 0)

static void DefaultInitializeTextureSlots(TRIGEN_HANDLE Trigen_Session session) {
    assert(session != nullptr);

    if(session == nullptr) {
        return;
    }

    Trigen_Texture_Slot_Descriptor slotBaseColor = { Trigen_PixFmt_RGB888 };
    TEXSLOTDESC_INIT_RGB888_BLACK(slotBaseColor);
    session->_texSlots.emplace_back(slotBaseColor);

    Trigen_Texture_Slot_Descriptor slotNormal = { Trigen_PixFmt_RGB888 };
    TEXSLOTDESC_INIT_RGB888(slotNormal, 128, 128, 255);
    session->_texSlots.emplace_back(slotNormal);

    Trigen_Texture_Slot_Descriptor slotHeightMap = { Trigen_PixFmt_RGB888 };
    TEXSLOTDESC_INIT_RGB888_BLACK(slotHeightMap);
    session->_texSlots.emplace_back(slotHeightMap);

    Trigen_Texture_Slot_Descriptor slotRoughness = { Trigen_PixFmt_RGB888 };
    TEXSLOTDESC_INIT_RGB888_BLACK(slotRoughness);
    session->_texSlots.emplace_back(slotRoughness);

    Trigen_Texture_Slot_Descriptor slotAO = { Trigen_PixFmt_RGB888 };
    TEXSLOTDESC_INIT_RGB888_BLACK(slotAO);
    session->_texSlots.emplace_back(slotAO);

    session->_inputTextures.resize(session->_texSlots.size());

    session->_texSlotsLocked = true;
}

static void
recompress(PSP::Mesh &mesh) {
    std::vector<glm::vec3> flatPositions;
    std::vector<glm::vec3> flatNormals;
    std::vector<glm::vec2> flatTexcoords;

    for (size_t i = 0; i < mesh.elements.size(); i++) {
        flatPositions.emplace_back(mesh.position[mesh.elements[i]]);
        flatNormals.emplace_back(mesh.normal[mesh.elements[i]]);
    }
    flatTexcoords = std::move(mesh.uv);

    TMC_Context ctx;
    TMC_Buffer bufPos, bufNormal, bufUV;
    TMC_Attribute attrPos, attrNormal, attrUV;

    TMC_CreateContext(&ctx, k_ETMCHint_None);
    TMC_SetIndexArrayType(ctx, k_ETMCType_UInt32);

    TMC_CreateBuffer(
        ctx, &bufPos, flatPositions.data(),
        flatPositions.size() * sizeof(flatPositions[0]));
    TMC_CreateBuffer(
        ctx, &bufNormal, flatNormals.data(),
        flatNormals.size() * sizeof(flatNormals[0]));
    TMC_CreateBuffer(
        ctx, &bufUV, flatTexcoords.data(),
        flatTexcoords.size() * sizeof(flatTexcoords[0]));

    TMC_CreateAttribute(
        ctx, &attrPos, bufPos, 3, k_ETMCType_Float32, sizeof(glm::vec3), 0);
    TMC_CreateAttribute(
        ctx, &attrNormal, bufNormal, 3, k_ETMCType_Float32, sizeof(glm::vec3), 0);
    TMC_CreateAttribute(
        ctx, &attrUV, bufUV, 2, k_ETMCType_Float32, sizeof(glm::vec2), 0);

    TMC_Compress(ctx, mesh.elements.size());

    void const *arrIndex;
    TMC_Size sizIndex, numIndex;
    TMC_GetIndexArray(ctx, &arrIndex, &sizIndex, &numIndex);

    mesh.elements.resize(numIndex);
    for (TMC_Size i = 0; i < numIndex; i++) {
        mesh.elements[i] = ((uint32_t *)arrIndex)[i];
    }

    void const *arrData;
    TMC_Size sizData, numData;

    TMC_GetDirectArray(ctx, attrPos, &arrData, &sizData);
    numData = sizData / sizeof(glm::vec3);
    mesh.position.resize(numData);
    memcpy(mesh.position.data(), arrData, sizData);

    TMC_GetDirectArray(ctx, attrNormal, &arrData, &sizData);
    numData = sizData / sizeof(glm::vec3);
    mesh.normal.resize(numData);
    memcpy(mesh.normal.data(), arrData, sizData);

    TMC_GetDirectArray(ctx, attrUV, &arrData, &sizData);
    numData = sizData / sizeof(glm::vec2);
    mesh.uv.resize(numData);
    memcpy(mesh.uv.data(), arrData, sizData);

    TMC_DestroyContext(ctx);
}

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
    s->_outputWidth = 512;
    s->_outputHeight = 512;

    if (!(params->flags & Trigen_F_UseGeneralTexturingAPI)) {
        DefaultInitializeTextureSlots(s);
    }

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

    assert(mesh->normals && mesh->positions && mesh->indices);
    if (mesh->normals == nullptr || mesh->positions == nullptr || mesh->indices == nullptr) {
        return Trigen_InvalidMesh;
    }

    assert(mesh->normal_count == mesh->position_count);
    if (mesh->normal_count != mesh->position_count) {
        return Trigen_InvalidMesh;
    }

    sb::Mesh_Collider sbCollider;
    sbCollider.triangle_count = mesh->triangle_count;
    sbCollider.indices = mesh->indices;
    sbCollider.num_positions = mesh->position_count;
    sbCollider.positions = mesh->positions;
    sbCollider.num_normals = mesh->normal_count;
    sbCollider.normals = mesh->normals;

    auto matOrient = glm::mat4_cast(glm::quat(transform->orientation[0], transform->orientation[1], transform->orientation[2], transform->orientation[3]));
    auto matTransform = glm::translate(matOrient, glm::vec3(transform->position[0], transform->position[1], transform->position[2]));

    sbCollider.transform = matTransform;

    sb::ISoftbody_Simulation::Collider_Handle handle;
    if (!session->simulation->add_collider(handle, &sbCollider)) {
        return Trigen_Failure;
    }

    session->colliders.push_front({ session->simulation.get(), handle });
    *collider = &session->colliders.front();

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

    auto matTranslate = glm::translate(
        glm::mat4(1.0f),
        glm::vec3(
            transform->position[0], transform->position[1],
            transform->position[2]));
    auto matOrient = glm::mat4_cast(glm::quat(transform->orientation[0], transform->orientation[1], transform->orientation[2], transform->orientation[3]));

    auto matTransform = matTranslate * matOrient;

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
    assert(scale > 0.0f);

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
    
    // PSP generates UVs for each element, so we need to recompress the mesh
    recompress(*(session->_pspMesh));

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

    assert(
        mesh.position.size() == mesh.normal.size()
        && mesh.normal.size() == mesh.uv.size());

    outMesh->position_count = mesh.position.size();
    outMesh->normal_count = mesh.normal.size();
    outMesh->triangle_count = mesh.elements.size() / 3;

    outMesh->positions = new float[mesh.position.size() * 3];
    outMesh->normals = new float[mesh.normal.size() * 3];
    outMesh->uvs = new float[mesh.uv.size() * 2];
    outMesh->indices = new uint64_t[mesh.elements.size() * 3];

    memcpy((void*)outMesh->positions, mesh.position.data(), outMesh->position_count * sizeof(glm::vec3));
    memcpy((void*)outMesh->normals, mesh.normal.data(), outMesh->normal_count * sizeof(glm::vec3));
    memcpy((void*)outMesh->uvs, mesh.uv.data(), mesh.uv.size() * sizeof(glm::vec2));

    static_assert(sizeof(uint64_t) == sizeof(size_t), "size_t was not 8 bytes! Fix this memcpy:");
    memcpy((void *)outMesh->indices, mesh.elements.data(), mesh.elements.size() * sizeof(uint64_t));

    return Trigen_OK;
}

TRIGEN_RETURN_CODE TRIGEN_API Trigen_Mesh_FreeMesh(
    TRIGEN_FREED TRIGEN_INOUT Trigen_Mesh *mesh) {
    if (mesh != nullptr) {
        delete[] mesh->indices;
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
        !(0 <= kind && kind < session->_inputTextures.size()) ||
        texture == nullptr ||
        texture->image == nullptr) {
        return Trigen_InvalidArguments;
    }

    assert(kind < session->_inputTextures.size());
    assert(kind < session->_texSlots.size());

    auto &tex = session->_inputTextures[kind];
    auto &desc = session->_texSlots[kind];

    int pixelSize;

    switch(desc.pixel_format) {
        case Trigen_PixFmt_RGB888:
        pixelSize = 3;
        break;
        case Trigen_PixFmt_MAX:
        std::abort();
        break;
    }

    // TODO: assuming no padding here
    auto size = texture->width * texture->height * pixelSize;

    auto data = std::make_unique<uint8_t[]>(size);
    memcpy(data.get(), texture->image, size);
    tex.data = std::move(data);
    tex.info.buffer = tex.data.get();
    tex.info.width = texture->width;
    tex.info.height = texture->height;

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

    session->_outputWidth = width;
    session->_outputHeight = height;

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

    PSP::Input_Material inputMaterial;
    std::vector<PSP::Input_Texture> placeholderTextureDescriptors;
    inputMaterial.reserve(session->_texSlots.size());

    for(size_t slot = 0; slot < session->_texSlots.size(); slot++) {
        auto &desc = session->_texSlots[slot];
        auto &tex = session->_inputTextures[slot];

        if(tex.data == nullptr) {
            PSP::Input_Texture placeholder;
            placeholder.buffer = &desc.defaultPixel;
            placeholder.width = placeholder.height = 1;
            inputMaterial.emplace_back(placeholder);
        } else {
            inputMaterial.emplace_back(tex.info);
        }
    }

    auto paint_input = PSP::Paint_Input {
        &session->_pspMesh.value(),
        inputMaterial,
        session->_outputWidth,
        session->_outputHeight
    };

    session->_outputMaterial = PSP::paint(paint_input);

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

    // TODO: check that the number of configured textures and output textures match
    // :GeneralTexturingAPI
    if (session->_outputMaterial.size() == 0) {
        return Trigen_NotReady;
    }

    PSP::Output_Texture const &tex = session->_outputMaterial[kind];

    if (tex.buffer == nullptr) {
        assert(!"Output material is complete, but the texture was empty");
        return Trigen_NotReady;
    }

    texture->width = tex.width;
    texture->height = tex.height;
    texture->image = tex.buffer.get();

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

TRIGEN_RETURN_CODE TRIGEN_API Trigen_CreateTextureSlot(
    TRIGEN_HANDLE Trigen_Session session,
    TRIGEN_OUT Trigen_Texture_Kind *slotHandle,
    TRIGEN_IN Trigen_Texture_Slot_Descriptor *descriptor) {
    if (session == nullptr ||
        slotHandle == nullptr ||
        descriptor == nullptr) {
        return Trigen_InvalidArguments;
    }

    if(session->_texSlotsLocked) {
        return Trigen_FunctionIsUnavailable;
    }

    if (descriptor->pixel_format >= Trigen_PixFmt_MAX) {
        return Trigen_InvalidArguments;
    }

    auto idx = session->_texSlots.size();
    session->_texSlots.emplace_back(*descriptor);
    session->_inputTextures.resize(session->_texSlots.size());
    *slotHandle = (Trigen_Texture_Kind)idx;

    return Trigen_OK;
}

TRIGEN_RETURN_CODE TRIGEN_API
Trigen_RegenerateFoliage(TRIGEN_HANDLE Trigen_Session session) {
    if (session == nullptr) {
        return Trigen_InvalidArguments;
    }

    auto foliageGenerator = make_foliage_generator(session->simulation);
    if (!foliageGenerator->generate()) {
        return Trigen_Failure;
    }

    session->foliageGenerator = std::move(foliageGenerator);

    // Convert indices from u32 to u64
    auto elements = session->foliageGenerator->elements();
    session->foliageIndexBuffer.resize(foliageGenerator->numElements());
    for (uint32_t i = 0; i < session->foliageGenerator->numElements(); i++) {
        session->foliageIndexBuffer[i] = static_cast<tg_u64>(elements[i]);
    }

    return Trigen_OK;
}

TRIGEN_RETURN_CODE TRIGEN_API
Trigen_GetFoliageMesh(
    TRIGEN_HANDLE Trigen_Session session,
    TRIGEN_OUT Trigen_Mesh* mesh) {
    if (session == nullptr || mesh == nullptr) {
        return Trigen_InvalidArguments;
    }

    if (session->foliageGenerator == nullptr) {
        return Trigen_NotReady;
    }

    mesh->positions = reinterpret_cast<tg_f32 const *>(session->foliageGenerator->positions());
    mesh->normals = reinterpret_cast<tg_f32 const *>(session->foliageGenerator->normals());
    mesh->uvs = reinterpret_cast<tg_f32 const *>(session->foliageGenerator->texcoords());
    mesh->position_count = session->foliageGenerator->numVertices();
    mesh->normal_count = session->foliageGenerator->numVertices();
    mesh->indices = session->foliageIndexBuffer.data();
    mesh->triangle_count = session->foliageIndexBuffer.size() / 3;

    return Trigen_OK;
}
}

