// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"
#include "collimp.h"
#include <vector>
#include <tiny_obj_loader.h>
#include "world.h"
#include <trigen/mesh_compress.h>

struct Mesh_Data {
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texcoords;
    std::vector<uint32_t> indices;
};

class Mesh_Collider_OBJ : public IMesh_Collider {
public:
    Mesh_Collider_OBJ(Mesh_Data &&meshData) :
        _meshData(std::move(meshData)) {
    }

    ~Mesh_Collider_OBJ() override = default;
private:
    std::optional<trigen::Collider> uploadToSimulation(trigen::Session &session) override {
        Trigen_Collider_Mesh descriptor = {};

        auto N = _meshData.indices.size() / 3;
        descriptor.triangle_count = N;

        std::vector<uint64_t> indices64;
        indices64.reserve(_meshData.indices.size());
        for (auto &idx : _meshData.indices) {
            indices64.emplace_back(idx);
        }

        descriptor.indices = indices64.data();

        descriptor.position_count = _meshData.positions.size();
        descriptor.positions = (float *)_meshData.positions.data();

        descriptor.normal_count = _meshData.normals.size();
        descriptor.normals = (float *)_meshData.normals.data();

        Trigen_Transform transform;
        memset(&transform, 0, sizeof(transform));
        transform.scale[0] = transform.scale[1] = transform.scale[2] = 1;
        transform.orientation[0] = 1.0f;

        try {
            return trigen::Collider::make(session, descriptor, transform);
        } catch(trigen::Exception const &) {
            return std::nullopt;
        }
    }

    topo::Model_ID uploadToRenderer(topo::IInstance *renderer) override {
        topo::Model_ID ret;
        topo::Model_Descriptor descriptor = {};

        std::vector<unsigned> elements;
        std::vector<glm::vec2> uvs;

        for (auto &element : _meshData.indices) {
            elements.push_back(element);
        }

        for (size_t i = 0; i < _meshData.indices.size(); i++) {
            uvs.push_back({ 0.0f, 0.0f });
        }

        descriptor.vertex_count = _meshData.positions.size();
        descriptor.vertices = (std::array<float, 3>*)_meshData.positions.data();
        descriptor.normals = (std::array<float, 3>*)_meshData.normals.data();
        descriptor.uv = (std::array<float, 2>*)_meshData.texcoords.data();

        descriptor.elements = elements.data();
        descriptor.element_count = elements.size();

        if (!renderer->CreateModel(&ret, &descriptor)) {
            std::abort();
        }

        return ret;
    }

    topo::Transform transform() const override {
        topo::Transform ret = {};
        ret.position = { 0, 0, 0 };
        ret.rotation = { 1, 0, 0, 0 };
        ret.scale = { 1, 1, 1 };
        return ret;
    }

private:
    Mesh_Data _meshData;
};

#if defined(NDEBUG)
#define CHK_TMC(expr)                                                          \
    if ((expr) != k_ETMCStatus_OK)                                             \
        std::abort();
#else
#define CHK_TMC(expr) assert((expr) == k_ETMCStatus_OK)
#endif

class Collider_Importer_OBJ : public ICollider_Importer {
public:
    ~Collider_Importer_OBJ() override = default;

private:
    std::vector<std::unique_ptr<IMesh_Collider>> loadFromFile(char const *path) override {
        std::vector<std::unique_ptr<IMesh_Collider>> ret;

        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string err;

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &err, path)) {
            return ret;
        }

        // We create a separate mesh collider for each shape/mesh in
        // the model
        for(int sidx = 0; sidx < shapes.size(); sidx++) {
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

            Mesh_Data meshData = { std::move(positions), std::move(normals),
                                   std::move(texcoords), std::move(indices) };

            ret.emplace_back(
                std::make_unique<Mesh_Collider_OBJ>(std::move(meshData)));

            CHK_TMC(TMC_DestroyContext(ctx));
        }

        return ret;
    }

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

    template<size_t NumComp>
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
};

std::unique_ptr<ICollider_Importer> makeObjColliderImporter() {
    auto importer = std::make_unique<Collider_Importer_OBJ>();

    return importer;
}
