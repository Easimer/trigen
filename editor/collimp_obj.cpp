// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"
#include "collimp.h"
#include <vector>
#include <tiny_obj_loader.h>
#include "world.h"

/*
 * Common data for meshes loaded from the same obj file
 */
struct Model_Data {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
};

struct Mesh_Data {
    std::vector<uint64_t> vertex_indices;
    std::vector<uint64_t> normal_indices;
};

class Mesh_Collider_OBJ : public IMesh_Collider {
public:
    Mesh_Collider_OBJ(std::shared_ptr<Model_Data> &modelData, Mesh_Data &&meshData, int shapeIndex) :
        _modelData(modelData),
        _meshData(std::move(meshData)),
        _shapeIndex(shapeIndex) {
    }

    ~Mesh_Collider_OBJ() override = default;
private:
    sb::ISoftbody_Simulation::Collider_Handle uploadToSimulation(sb::ISoftbody_Simulation *sim) override {
        sb::ISoftbody_Simulation::Collider_Handle handle;
        sb::Mesh_Collider descriptor = {};

        auto &indices = _modelData->shapes[_shapeIndex].mesh.indices;
        auto N = indices.size() / 3;
        descriptor.transform = glm::mat4(1.0f);
        descriptor.triangle_count = N;

        descriptor.vertex_indices = _meshData.vertex_indices.data();
        descriptor.normal_indices = _meshData.normal_indices.data();

        descriptor.position_count = _modelData->attrib.vertices.size();
        descriptor.positions = (float *)_modelData->attrib.vertices.data();

        descriptor.normal_count = _modelData->attrib.normals.size();
        descriptor.normals = (float *)_modelData->attrib.normals.data();


        if (!sim->add_collider(handle, &descriptor)) {
            std::abort();
        }

        return handle;
    }

    gfx::Model_ID uploadToRenderer(gfx::IRenderer *renderer) override {
        gfx::Model_ID ret;
        gfx::Model_Descriptor descriptor = {};

        std::vector<unsigned> elements;
        auto vertices = (std::array<float, 3>*)_modelData->attrib.vertices.data();
        std::vector<glm::vec2> uvs;

        for (auto &element : _meshData.vertex_indices) {
            elements.push_back(element);
        }

        for (size_t i = 0; i < _meshData.vertex_indices.size(); i++) {
            uvs.push_back({ 0.0f, 0.0f });
        }

        descriptor.vertices = vertices;
        descriptor.vertex_count = _modelData->attrib.vertices.size();
        descriptor.elements = elements.data();
        descriptor.element_count = elements.size();
        descriptor.uv = (std::array<float, 2>*)uvs.data();

        if (!renderer->create_model(&ret, &descriptor)) {
            std::abort();
        }

        return ret;
    }

    gfx::Transform transform() const override {
        gfx::Transform ret;
        ret.position = { 0, 0, 0 };
        ret.rotation = { 1, 0, 0, 0 };
        ret.scale = { 1, 1, 1 };
        return ret;
    }

private:
    std::shared_ptr<Model_Data> _modelData;
    Mesh_Data _meshData;
    int _shapeIndex;
};

class Collider_Importer_OBJ : public ICollider_Importer {
public:
    ~Collider_Importer_OBJ() override = default;

private:
    std::vector<std::unique_ptr<IMesh_Collider>> loadFromFile(char const *path) override {
        std::vector<std::unique_ptr<IMesh_Collider>> ret;

        auto modelData = std::make_shared<Model_Data>();
        std::string err;

        if (!tinyobj::LoadObj(&modelData->attrib, &modelData->shapes, &modelData->materials, &err, path)) {
            return ret;
        }

        // We create a separate mesh collider for each shape/mesh in
        // the model
        for(int sidx = 0; sidx < modelData->shapes.size(); sidx++) {
            auto &shape = modelData->shapes[sidx];
            Mesh_Data meshData;

            auto& vertex_indices = meshData.vertex_indices;
            auto& normal_indices = meshData.normal_indices;
            for (auto &index : shape.mesh.indices) {
                vertex_indices.push_back((uint64_t)index.vertex_index);
                normal_indices.push_back((uint64_t)index.normal_index);
            }

            ret.emplace_back(std::make_unique<Mesh_Collider_OBJ>(modelData, std::move(meshData), sidx));
        }

        return ret;
    }
};

std::unique_ptr<ICollider_Importer> makeObjColliderImporter() {
    auto importer = std::make_unique<Collider_Importer_OBJ>();

    return importer;
}
