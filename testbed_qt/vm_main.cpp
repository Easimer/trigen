// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: main window viewmodel implementation
//

#include "common.h"
#include "vm_main.h"
#include <map>

// NOTE: we don't need this define because the static library 'objscan' already
// contains an implementation.
// #define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

using Collider_Handle = sb::ISoftbody_Simulation::Collider_Handle;

// Identifies a specific Mesh_Data in _mesh_data
using Mesh_Index = int;

// Identifies a specific shape in a Mesh_Data
using Shape_Index = int;

// Identifies a specific shape in a specific mesh
using Mesh_Shape_Index = std::pair<Mesh_Index, Shape_Index>;

struct Mesh_Data {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
};

class Viewmodel_Main : public IViewmodel_Main {
public:
    Viewmodel_Main(sb::ISoftbody_Simulation *sim) : _simulation(sim) {
    }

private:
    bool add_mesh_collider(char const *path, std::string &err_msg) override {
        std::string warn, err;
        Mesh_Data mesh;

        if (!tinyobj::LoadObj(&mesh.attrib, &mesh.shapes, &mesh.materials, &err, path)) {
            return false;
        }

        glm::mat4 transform(1.0f);

        Mesh_Index midx = _mesh_data.size();

        for(Shape_Index sidx = 0; sidx < mesh.shapes.size(); sidx++) {
            auto &shape = mesh.shapes[sidx];

            sb::Mesh_Collider coll;

            fill_out_collider_struct(&coll, mesh, shape);

            Collider_Handle handle;
            _simulation->add_collider(handle, &coll);

            _mesh_colliders[handle] = { midx, sidx };
        }

        _mesh_data.push_back(std::move(mesh));

        return true;
    }

    int num_mesh_colliders() override {
        return _mesh_colliders.size();
    }

    bool mesh_collider(sb::Mesh_Collider *mesh, int index) override {
        assert(mesh != NULL || index >= 0);
        if (mesh == NULL || index < 0) {
            return false;
        }

        // is there a way to lookup by index?
        auto it = _mesh_colliders.cbegin();

        while (index > 0) {
            ++it;
            index--;

            if (it == _mesh_colliders.cend()) {
                return false;
            }
        }

        auto [handle, msidx] = *it;

        auto &mesh_data = _mesh_data[msidx.first];
        auto &shape = mesh_data.shapes[msidx.second];

        fill_out_collider_struct(mesh, mesh_data, shape);

        return true;
    }

    void fill_out_collider_struct(sb::Mesh_Collider *collider, Mesh_Data &mesh, tinyobj::shape_t &shape) {
        auto &indices = shape.mesh.indices;
        auto N = indices.size() / 3;
        collider->transform = Mat4(1.0f);
        collider->triangle_count = N;

        // need to copy the vertex indices because they are in a AoS layout
        // but we need a contiguous integer array
        std::vector<uint64_t> vertex_indices;
        std::vector<uint64_t> normal_indices;
        for (auto &index : indices) {
            vertex_indices.push_back((uint64_t)index.vertex_index);
            normal_indices.push_back((uint64_t)index.normal_index);
        }
        collider->vertex_indices = vertex_indices.data();
        collider->normal_indices = normal_indices.data();

        collider->position_count = mesh.attrib.vertices.size();
        collider->positions = (float *)mesh.attrib.vertices.data();

        collider->normal_count = mesh.attrib.normals.size();
        collider->normals = (float *)mesh.attrib.normals.data();
    }

private:
    sb::ISoftbody_Simulation *_simulation;
    std::vector<Mesh_Data> _mesh_data;
    std::map<Collider_Handle, Mesh_Shape_Index> _mesh_colliders;
};

Unique_Ptr<IViewmodel_Main> make_viewmodel_main(sb::ISoftbody_Simulation *simulation) {
    return std::make_unique<Viewmodel_Main>(simulation);
}
