// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"
#include "fbx_standard_surface.h"
#include <cassert>

static void TrySetDefaultValue(char const *name, FbxProperty &prop) {
#define SET_DEFAULT_VALUE_IF_NAME(pname, value) if(strcmp(name, pname) == 0) { prop.Set(value); return; }
    SET_DEFAULT_VALUE_IF_NAME("normalCamera", FbxDouble3(1, 1, 1));
    SET_DEFAULT_VALUE_IF_NAME("base", 0.8f);
    SET_DEFAULT_VALUE_IF_NAME("baseColor", FbxDouble3(1, 1, 1));
    SET_DEFAULT_VALUE_IF_NAME("specular", 1.0f);
    SET_DEFAULT_VALUE_IF_NAME("specularColor", FbxDouble3(1, 1, 1));
    SET_DEFAULT_VALUE_IF_NAME("specularRoughness", 0.2f);
    SET_DEFAULT_VALUE_IF_NAME("specularIOR", 1.5f);
    SET_DEFAULT_VALUE_IF_NAME("transmissionColor", FbxDouble3(1, 1, 1));
    SET_DEFAULT_VALUE_IF_NAME("subsurfaceColor", FbxDouble3(1, 1, 1));
    SET_DEFAULT_VALUE_IF_NAME("subsurfaceRadius", FbxDouble3(1, 1, 1));
    SET_DEFAULT_VALUE_IF_NAME("subsurfaceScale", 1.0f);
    SET_DEFAULT_VALUE_IF_NAME("subsurfaceType", 1);
    SET_DEFAULT_VALUE_IF_NAME("sheenColor", FbxDouble3(1, 1, 1));
    SET_DEFAULT_VALUE_IF_NAME("sheenRoughness", 0.3f);
    SET_DEFAULT_VALUE_IF_NAME("coatColor", FbxDouble3(1, 1, 1));
    SET_DEFAULT_VALUE_IF_NAME("coatRoughness", 0.1f);
    SET_DEFAULT_VALUE_IF_NAME("coatIOR", 1.5f);
    SET_DEFAULT_VALUE_IF_NAME("thinFilmIOR", 1.5f);
    SET_DEFAULT_VALUE_IF_NAME("emissionColor", FbxDouble3(1, 1, 1));
    SET_DEFAULT_VALUE_IF_NAME("opacity", FbxDouble3(1, 1, 1));
    SET_DEFAULT_VALUE_IF_NAME("internalReflections", true);
    SET_DEFAULT_VALUE_IF_NAME("indirectDiffuse", 1.0f);
    SET_DEFAULT_VALUE_IF_NAME("indirectSpecular", 1.0f);
}

#define TRY_SET_DEFAULT_VALUE(name, type) TrySetDefaultValue(#name, _prop_##name)

FbxBindingTable *fbx_standard_surface_create_binding_table(FbxImplementation *impl) {
    assert(impl != nullptr);

    FbxBindingTable *table = nullptr;
    if (impl != nullptr) {
        table = impl->AddNewTable("root", "shader");
        assert(table != nullptr);

        if (table != nullptr) {
            impl->RootBindingName = "root";

            table->TargetName.Set("root");
            table->TargetType.Set("shader");

            FBX_FOR_ALL_PROPERTY(FBX_ADD_BINDING_ENTRY);
        }
    }

    return table;
}

FbxImplementation *fbx_standard_surface_create_implementation(FbxScene *scene) {
    auto impl = FbxImplementation::Create(scene, "implStandardSurface");
    assert(impl != nullptr);
    
    if (impl != nullptr) {
        impl->Language.Set("AiOSL");
        impl->RenderAPI.Set("ARNOLD_SHADER_ID");
        impl->RenderAPIVersion.Set("4.0");
    }

    return impl;
}

FbxArnoldStandardSurface::FbxArnoldStandardSurface(FbxScene *scene, char const *name) : _material(nullptr) {
    assert(scene != nullptr);
    assert(name != nullptr);

    auto surf = FbxSurfaceMaterial::Create(scene, name);
    assert(surf != nullptr);

    _material = surf;

    surf->ShadingModel.Set("");
    FbxProperty::Create(surf, FbxStringDT, "ShadingModel");

    auto root = FbxProperty::Create(surf, FbxCompoundDT, "Maya");
    FbxPropertyT<FbxInt> typeId = FbxProperty::Create(root, FbxIntDT, "TypeId");
    typeId.Set(1138001);

    FBX_FOR_ALL_PROPERTY(CREATE_PROPERTY);
    FBX_FOR_ALL_PROPERTY(TRY_SET_DEFAULT_VALUE);
}
