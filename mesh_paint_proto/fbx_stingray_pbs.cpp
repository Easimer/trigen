// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "fbx_stingray_pbs.h"

#define PROP_NAME_MAYA "Maya"

FBXSDK_OBJECT_IMPLEMENT(StingrayPBS);

char const *StingrayPBS::GetTypeName() const {
    return "StingrayPBS";
}

FbxProperty StingrayPBS::GetProperty(int pId) {
    // TODO(danielm): do we need this?
    FbxProperty dummy;
    switch (pId) {
    case ePbrMaya: return FindProperty(PROP_NAME_MAYA);
    }

    return dummy;
}

void StingrayPBS::ConstructProperties(bool pForceSet) {
    ParentClass::ConstructProperties(pForceSet);

    auto root = FbxProperty::Create(this, FbxCompoundDT, PROP_NAME_MAYA, "MayaLabel");

    this->TypeId = FbxProperty::Create(root, FbxIntDT, "TypeId");
    this->UseNormalMap = FbxProperty::Create(root, FbxFloatDT, "use_normal_map");
    this->NormalMap = FbxProperty::Create(root, FbxDouble3DT, "TEX_normal_map");

    this->GlobalDiffuseCube = FbxProperty::Create(root, FbxDouble3DT, "TEX_global_diffuse_cube");
    this->GlobalSpecularCube = FbxProperty::Create(root, FbxDouble3DT, "TEX_global_specular_cube");
    this->BrdfLut = FbxProperty::Create(root, FbxDouble3DT, "TEX_brdf_lut");

    this->UvOffset = FbxProperty::Create(root, FbxFloatDT, "uv_offset");
    this->UvScale = FbxProperty::Create(root, FbxFloatDT, "uv_scale");

    this->UseNormalMap = FbxProperty::Create(root, FbxFloatDT, "use_normal_map");
    this->NormalMap = FbxProperty::Create(root, FbxDouble3DT, "TEX_normal_map");

    this->UseColorMap = FbxProperty::Create(root, FbxFloatDT, "use_color_map");
    this->ColorMap = FbxProperty::Create(root, FbxDouble3DT, "TEX_color_map");
    this->BaseColor = FbxProperty::Create(root, FbxDouble3DT, "base_color");

    this->UseMetallicMap = FbxProperty::Create(root, FbxFloatDT, "use_metallic_map");
    this->MetallicMap = FbxProperty::Create(root, FbxDouble3DT, "TEX_metallic_map");
    this->Metallic = FbxProperty::Create(root, FbxFloatDT, "metallic");

    this->UseRoughnessMap = FbxProperty::Create(root, FbxFloatDT, "use_roughness_map");
    this->RoughnessMap = FbxProperty::Create(root, FbxDouble3DT, "TEX_roughness_map");
    this->Roughness = FbxProperty::Create(root, FbxFloatDT, "roughness");

    this->UseEmissiveMap = FbxProperty::Create(root, FbxFloatDT, "use_emissive_map");
    this->EmissiveMap = FbxProperty::Create(root, FbxDouble3DT, "TEX_emissive_map");
    this->Emissive = FbxProperty::Create(root, FbxDouble3DT, "emissive");
    this->EmissiveIntensity = FbxProperty::Create(root, FbxFloatDT, "");

    this->UseAoMap = FbxProperty::Create(root, FbxFloatDT, "use_ao_map");
    this->AoMap = FbxProperty::Create(root, FbxDouble3DT, "TEX_ao_map");
}

void StingrayPBS::Construct(FbxObject const *from) {
    ParentClass::Construct(from);

    ShadingModel.Set("unknown");
    TypeId.Set(1166017);
    BaseColor.Set({ 0.5, 0.5, 0.5 });
}

void StingrayPBS::Destruct(bool bRecursive) {
    ParentClass::Destruct(bRecursive);
}

// REMOVE:
FbxSurfaceLambert *asd;
