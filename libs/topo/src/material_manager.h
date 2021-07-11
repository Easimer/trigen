// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#pragma once

#include <list>

#include <topo.h>

namespace topo {
struct Material_Lit {
    Texture_ID diffuse;
    Texture_ID normal;
};

struct Material_Unlit {
    Texture_ID diffuse;
};

struct Material_Transparent {
    Texture_ID diffuse;
};

struct Material_Solid_Color {
    float color[3];
};

enum Material_Type {
    MAT_LIT,
    MAT_UNLIT,
    MAT_UNLIT_TRANSPARENT,
    MAT_SOLID_COLOR,
};

class Material_Manager {
public:
    bool
    CreateUnlitMaterial(Material_ID *outHandle, Texture_ID diffuse);

    bool
    CreateUnlitTransparentMaterial(Material_ID *outHandle, Texture_ID diffuse);

    bool
    CreateLitMaterial(
        Material_ID *outHandle,
        Texture_ID diffuse,
        Texture_ID normal);

    bool
    CreateSolidColorMaterial(Material_ID *outHandle, glm::vec3 const &color);

    void
    DestroyMaterial(Material_ID material);

    Material_Type
    GetType(Material_ID material);

    void *
    GetMaterialData(Material_ID material);

private:
    using Material_Tag = Material_Type;

    struct Tagged_Material {
        Material_Tag tag;

        union {
            Material_Lit lit;
            Material_Unlit unlit;
            Material_Transparent unlit_transparent;
            Material_Solid_Color solid;
        };
    };

    std::list<Tagged_Material> _materials;
};
}
