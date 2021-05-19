// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <psp/psp.h>

bool fbx_try_save(char const *path, PSP::Mesh const *mesh, PSP::Material const *material);

struct Mesh_Export_Mesh {
    // Number of triangles in this mesh.
    // The number of elements in the index buffer should be three
    // times this number.
    size_t triangle_count;

    // Pointer to the vertex index buffer.
    uint64_t const *vertex_indices;
    // Pointer to the normal index buffer.
    uint64_t const *normal_indices;

    // Number of elements in the position buffer.
    // This should be at least `max(vertex_indices)+1`.
    size_t position_count;
    // Pointer to the position vector buffer.
    // Assumed to be in the following format: XYZ XYZ XYZ
    float const *positions;
    // Pointer to the UV buffer.
    // Assumed to be in the following format: UV UV UV
    float const *uvs;

    // Number of elements in the normal buffer.
    // This should be at least `max(normal_indices)+1`.
    size_t normal_count;
    // Pointer to the normal vector buffer.
    // Assumed to be in the following format: XYZ XYZ XYZ
    float const *normals;
};

struct Mesh_Export_Texture {
    void const *image;
    int width, height;
};

struct Mesh_Export_Material {
    Mesh_Export_Texture base;
    Mesh_Export_Texture normal;
    Mesh_Export_Texture height;
    Mesh_Export_Texture roughness;
    Mesh_Export_Texture ao;
};

bool fbx_try_save(char const *path, Mesh_Export_Mesh const &mesh, Mesh_Export_Material const &material);