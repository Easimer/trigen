// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <psp/psp.h>
#include <trigen.h>

bool fbx_try_save(char const *path, PSP::Mesh const *mesh, PSP::Material const *material);

struct Trigen_Material {
    Trigen_Texture const *base;
    Trigen_Texture const *normal;
    Trigen_Texture const *height;
    Trigen_Texture const *roughness;
    Trigen_Texture const *ao;
};

bool fbx_try_save(char const *path, Trigen_Mesh const &mesh, Trigen_Material const &material);