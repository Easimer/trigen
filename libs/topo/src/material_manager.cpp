// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#include "material_manager.h"

namespace topo {
bool
Material_Manager::CreateUnlitMaterial(
    Material_ID *outHandle,
    Texture_ID diffuse) {
    if (outHandle == nullptr || diffuse == nullptr) {
        return false;
    }

    Tagged_Material mat;
    mat.tag = MAT_UNLIT;
    mat.unlit.diffuse = diffuse;

    _materials.push_front(mat);
    *outHandle = &_materials.front();

    return true;
}

bool
Material_Manager::CreateUnlitTransparentMaterial(
    Material_ID *outHandle,
    Texture_ID diffuse) {

    if (outHandle == nullptr || diffuse == nullptr) {
        return false;
    }

    Tagged_Material mat;
    mat.tag = MAT_UNLIT_TRANSPARENT;
    mat.unlit_transparent.diffuse = diffuse;

    _materials.push_front(mat);
    *outHandle = &_materials.front();
    return true;
}

bool
Material_Manager::CreateLitMaterial(
    Material_ID *outHandle,
    Texture_ID diffuse,
    Texture_ID normal) {
    if (outHandle == nullptr || diffuse == nullptr || normal == nullptr) {
        return false;
    }

    Tagged_Material mat;
    mat.tag = MAT_LIT;
    mat.lit.diffuse = diffuse;
    mat.lit.normal = normal;

    _materials.push_front(mat);
    *outHandle = &_materials.front();

    return true;
}

bool
Material_Manager::CreateSolidColorMaterial(
    Material_ID *outHandle,
    glm::vec3 const &color) {
    if (outHandle == nullptr) {
        return false;
    }

    Tagged_Material mat;
    mat.tag = MAT_SOLID_COLOR;
    mat.solid.color[0] = color[0];
    mat.solid.color[1] = color[1];
    mat.solid.color[2] = color[2];

    _materials.push_front(mat);
    *outHandle = &_materials.front();

    return true;
}

void
Material_Manager::DestroyMaterial(Material_ID id) {
    if (id == nullptr) {
        return;
    }

    std::remove_if(_materials.begin(), _materials.end(), [&](Tagged_Material const &t) {
        return &t == id;
    });
}

Material_Type
Material_Manager::GetType(Material_ID material) {
    if (material != nullptr) {
        return ((Tagged_Material *)material)->tag;
    }

    std::abort();
}

void *
Material_Manager::GetMaterialData(Material_ID material) {
    if (material != nullptr) {
        return &((Tagged_Material *)material)->lit;
    }

    std::abort();
}
}
