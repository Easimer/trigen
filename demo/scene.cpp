// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "scene.h"
#include <memory>
#include <algorithm>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include <trigen/mesh_compress.h>

#if defined(NDEBUG)
#define CHK_TMC(expr)                                                          \
    if ((expr) != k_ETMCStatus_OK)                                             \
        std::abort();
#else
#define CHK_TMC(expr) assert((expr) == k_ETMCStatus_OK)
#endif

static void
flatten_mesh(
    tinyobj::attrib_t const &attrib,
    tinyobj::shape_t const &shape,
    std::vector<glm::vec3> &positions,
    std::vector<glm::vec3> &normals,
    std::vector<glm::vec2> &texcoords) {
    auto const N = shape.mesh.indices.size();
    positions.clear();
    normals.clear();
    positions.reserve(N);
    normals.reserve(N);

    auto aVertices = (glm::vec3 const *)attrib.vertices.data();
    auto aNormals = (glm::vec3 const *)attrib.normals.data();
    auto aTexcoords = (glm::vec2 const *)attrib.texcoords.data();

    for (size_t i = 0; i < N; i++) {
        auto idxPos = shape.mesh.indices[i].vertex_index;
        auto idxNormal = shape.mesh.indices[i].normal_index;
        auto idxTexcoord = shape.mesh.indices[i].texcoord_index;
        positions.emplace_back(aVertices[idxPos]);
        normals.emplace_back(aNormals[idxNormal]);
        texcoords.emplace_back(aTexcoords[idxTexcoord]);
    }
}

template <size_t NumComp>
static std::vector<glm::vec<NumComp, float>>
attribute_to_vector(TMC_Context ctx, TMC_Attribute attr) {
    std::vector<glm::vec<NumComp, float>> ret;
    void const *data;
    TMC_Size size;
    CHK_TMC(TMC_GetDirectArray(ctx, attr, &data, &size));
    TMC_Size count = size / sizeof(glm::vec<NumComp, float>);
    ret.resize(count);
    auto *dataVec = (glm::vec<NumComp, float> const *)data;

    for (TMC_Size i = 0; i < count; i++) {
        ret[i] = dataVec[i];
    }

    return ret;
}

static std::vector<uint32_t>
indices_to_vector(TMC_Context ctx) {
    void const *data;
    TMC_Size size, count;

    CHK_TMC(TMC_GetIndexArray(ctx, &data, &size, &count));
    auto *indices = (uint32_t const *)data;

    std::vector<uint32_t> ret;
    ret.resize(count);

    memcpy(ret.data(), indices, size);

    return ret;
}

std::unique_ptr<Scene>
MakeScene_Basic_Cube(topo::IInstance *renderer, Trigen_Session simulation);

std::unique_ptr<Scene>
MakeScene(
    Scene::Kind kind,
    topo::IInstance *renderer,
    Trigen_Session simulation) {
    switch (kind) {
    case Scene::K_BASIC_CUBE:
        return MakeScene_Basic_Cube(renderer, simulation);
    }

    return nullptr;
}

std::vector<Scene::Collider>
Scene::LoadObjMeshCollider(
    topo::IInstance *renderer,
    Trigen_Session simulation,
    Trigen_Transform const &transform,
    char const *path) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err;
    std::vector<Collider> ret;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &err, path)) {
        return ret;
    }

    // We create a separate mesh collider for each shape/mesh in
    // the model
    for (int sidx = 0; sidx < shapes.size(); sidx++) {
        auto &shape = shapes[sidx];

        std::vector<glm::vec3> flatPositions;
        std::vector<glm::vec3> flatNormals;
        std::vector<glm::vec2> flatTexcoords;
        flatten_mesh(attrib, shape, flatPositions, flatNormals, flatTexcoords);

        TMC_Context ctx;
        TMC_Buffer bufPositions, bufNormals, bufTexcoords;
        TMC_Attribute attrPositions, attrNormals, attrTexcoords;

        CHK_TMC(TMC_CreateContext(&ctx, k_ETMCHint_None));
        CHK_TMC(TMC_SetIndexArrayType(ctx, k_ETMCType_UInt32));

        CHK_TMC(TMC_CreateBuffer(
            ctx, &bufPositions, flatPositions.data(),
            flatPositions.size() * sizeof(flatPositions[0])));
        CHK_TMC(TMC_CreateBuffer(
            ctx, &bufNormals, flatNormals.data(),
            flatNormals.size() * sizeof(flatNormals[0])));
        CHK_TMC(TMC_CreateBuffer(
            ctx, &bufTexcoords, flatTexcoords.data(),
            flatTexcoords.size() * sizeof(flatTexcoords[0])));

        CHK_TMC(TMC_CreateAttribute(
            ctx, &attrPositions, bufPositions, 3, k_ETMCType_Float32,
            sizeof(flatPositions[0]), 0));
        CHK_TMC(TMC_CreateAttribute(
            ctx, &attrNormals, bufNormals, 3, k_ETMCType_Float32,
            sizeof(flatNormals[0]), 0));
        CHK_TMC(TMC_CreateAttribute(
            ctx, &attrTexcoords, bufTexcoords, 2, k_ETMCType_Float32,
            sizeof(flatTexcoords[0]), 0));

        CHK_TMC(TMC_Compress(ctx, flatPositions.size()));

        auto positions = attribute_to_vector<3>(ctx, attrPositions);
        auto normals = attribute_to_vector<3>(ctx, attrNormals);
        auto texcoords = attribute_to_vector<2>(ctx, attrTexcoords);
        auto indices = indices_to_vector(ctx);

        topo::Model_ID hModel;
        topo::Model_Descriptor descriptor;
        descriptor.elements = indices.data();
        descriptor.element_count = indices.size();
        descriptor.normals = normals.data();
        descriptor.vertices = positions.data();
        descriptor.vertex_count = positions.size();
        descriptor.uv = texcoords.data();

        Trigen_Collider hCollider;
        Trigen_Collider_Mesh collDescriptor;
        std::vector<tg_u64> indicesU64;
        indicesU64.reserve(indices.size());
        std::transform(
            indices.cbegin(), indices.cend(), std::back_inserter(indicesU64),
            [&](uint32_t val) { return (tg_u64)val; });

        collDescriptor.indices = indicesU64.data();
        collDescriptor.triangle_count = indicesU64.size() / 3;
        collDescriptor.normals = (tg_f32*)normals.data();
        collDescriptor.normal_count = normals.size();
        collDescriptor.positions = (tg_f32*)positions.data();
        collDescriptor.position_count = positions.size();
        auto colliderOk = Trigen_CreateCollider(
                      &hCollider, simulation, &collDescriptor, &transform)
            == Trigen_OK;

        auto visualOk = renderer->CreateModel(&hModel, &descriptor);

        if (colliderOk && visualOk) {
            ret.emplace_back(hModel, hCollider);
        } else {
            if (colliderOk) {
                fprintf(stderr, "WHOOPS!, you can't remove a collider from a simulation at runtime!\n");
                // WORKAROUND: scale the collider into a point
                Trigen_Transform null = { {}, {}, { 0, 0, 0 } };
                Trigen_UpdateCollider(hCollider, &null);
            }
            if (visualOk) {
                renderer->DestroyModel(hModel);
            }
        }

        CHK_TMC(TMC_DestroyContext(ctx));
    }

    return ret;
}
