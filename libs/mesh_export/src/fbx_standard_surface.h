// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <fbxsdk.h>

#define FBX_DECLARE_PROPERTY(name, type) FbxPropertyT<type> name
#define FBX_CREATE_PROPERTY(name, type) this->name = FbxProperty::Create(root, type##DT, #name)
#define FBX_ADD_BINDING_ENTRY(name, type) \
{\
    auto &entry = table->AddNewEntry(); \
    entry.SetSource("Maya|" #name); \
    entry.SetDestination(#name); \
    entry.SetEntryType("FbxPropertyEntry", true); entry.SetEntryType("FbxSemanticEntry", false); \
}

#define FBX_FOR_ALL_PROPERTY(macro) \
    macro(outAlpha, FbxFloat); \
    macro(normalCamera, FbxDouble3); \
    macro(aiEnableMatte, FbxBool); \
    macro(aiMatteColor, FbxDouble3); \
    macro(aiMatteColorA, FbxFloat); \
    macro(base, FbxFloat); \
    macro(baseColor, FbxDouble3); \
    macro(diffuseRoughness, FbxFloat); \
    macro(specular, FbxFloat); \
    macro(specularColor, FbxDouble3); \
    macro(specularRoughness, FbxFloat); \
    macro(specularIOR, FbxFloat); \
    macro(specularAnisotropy, FbxFloat); \
    macro(specularRotation, FbxFloat); \
    macro(metalness, FbxFloat); \
    macro(transmission, FbxFloat); \
    macro(transmissionColor, FbxDouble3); \
    macro(transmissionDepth, FbxFloat); \
    macro(transmissionScatter, FbxDouble3); \
    macro(transmissionScatterAnisotropy, FbxFloat); \
    macro(transmissionDispersion, FbxFloat); \
    macro(transmissionExtraRoughness, FbxFloat); \
    macro(transmitAovs, FbxBool); \
    macro(subsurface, FbxFloat); \
    macro(subsurfaceColor, FbxDouble3); \
    macro(subsurfaceRadius, FbxDouble3); \
    macro(subsurfaceScale, FbxFloat); \
    macro(subsurfaceAnisotropy, FbxFloat); \
    macro(subsurfaceType, FbxEnum); \
    macro(sheen, FbxFloat); \
    macro(sheenColor, FbxDouble3); \
    macro(sheenRoughness, FbxFloat); \
    macro(thinWalled, FbxBool); \
    macro(tangent, FbxDouble3); \
    macro(coat, FbxFloat); \
    macro(coatColor, FbxDouble3); \
    macro(coatRoughness, FbxFloat); \
    macro(coatIOR, FbxFloat); \
    macro(coatAnisotropy, FbxFloat); \
    macro(coatRotation, FbxFloat); \
    macro(coatNormal, FbxDouble3); \
    macro(thinFilmThickness, FbxFloat); \
    macro(thinFilmIOR, FbxFloat); \
    macro(emission, FbxFloat); \
    macro(emissionColor, FbxDouble3); \
    macro(opacity, FbxDouble3); \
    macro(caustics, FbxBool); \
    macro(internalReflections, FbxBool); \
    macro(exitToBackground, FbxBool); \
    macro(indirectDiffuse, FbxFloat); \
    macro(indirectSpecular, FbxFloat); \
    macro(aovId1, FbxString); \
    macro(id1, FbxDouble3); \
    macro(aovId2, FbxString); \
    macro(id2, FbxDouble3); \
    macro(aovId3, FbxString); \
    macro(id3, FbxDouble3); \
    macro(aovId4, FbxString); \
    macro(id4, FbxDouble3); \
    macro(aovId5, FbxString); \
    macro(id5, FbxDouble3); \
    macro(aovId6, FbxString); \
    macro(id6, FbxDouble3); \
    macro(aovId7, FbxString); \
    macro(id7, FbxDouble3); \
    macro(aovId8, FbxString); \
    macro(id8, FbxDouble3); \
    macro(normalCameraFactor, FbxDouble);

#define DEFINE_GETTER(name, type) FbxPropertyT<type>& get_##name () { return _prop_##name; }
#define DEFINE_FIELD(name, type) FbxPropertyT<type> _prop_##name
#define CREATE_PROPERTY(name, type) _prop_##name = FbxProperty::Create(root, type##DT, #name)

class FbxArnoldStandardSurface {
public:
    FbxArnoldStandardSurface(FbxScene *scene, char const *name);

    FBX_FOR_ALL_PROPERTY(DEFINE_GETTER);

    FbxSurfaceMaterial *material() { return _material; }

private:
    FbxSurfaceMaterial *_material;

    FBX_FOR_ALL_PROPERTY(DEFINE_FIELD);
};

FbxImplementation *fbx_standard_surface_create_implementation(FbxScene *scene);
FbxBindingTable *fbx_standard_surface_create_binding_table(FbxImplementation *impl);