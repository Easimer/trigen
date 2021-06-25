// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

/*
 * Define MESH_EXPORT_PSP if you want to use the PSP:: version of fbx_try_save
 * before including this file;
 * also make sure that you link against the psp library, because mesh_export
 * doesn't.
 */

#ifdef MESH_EXPORT_PSP
#include <psp/psp.h>

bool fbx_try_save(char const *path, PSP::Mesh const *mesh, PSP::Material const *material);
#endif /* MESH_EXPORT_PSP */

#include <trigen.h>

struct Trigen_Material {
    Trigen_Texture const *base;
    Trigen_Texture const *normal;
    Trigen_Texture const *height;
    Trigen_Texture const *roughness;
    Trigen_Texture const *ao;
};

bool fbx_try_save(char const *path, Trigen_Mesh const &mesh, Trigen_Material const &material);

struct Export_Model {
    Trigen_Mesh mesh;
    Trigen_Material material;

    Trigen_Mesh foliageMesh;
    Trigen_Texture const *leafTexture;
};

bool fbx_try_save(char const *path, Export_Model const &model);