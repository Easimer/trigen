// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <fbxsdk.h>

class StingrayPBS : public FbxSurfaceMaterial {
	FBXSDK_OBJECT_DECLARE(StingrayPBS, FbxSurfaceMaterial);
public:
    typedef enum {
        ePbrMaya = 0,
        ePbrTypeId,

        ePbrTexGlobalDiffuseCube,
        ePbrTexGlobalSpecularCube,
        ePbrTexBrdfLut,

        ePbrUvOffset,
        ePbrUvScale,

        ePbrUseNormalMap,
        ePbrTexNormalMap,

        ePbrUseColorMap,
        ePbrTexColorMap,
        ePbrBaseColor,

        ePbrUseMetallicMap,
        ePbrTexMetallicMap,
        ePbrMetallic,

        ePbrUseRoughnessMap,
        ePbrTexRoughnessMap,
        ePbrRoughness,

        ePbrUseEmissiveMap,
        ePbrTexEmissiveMap,
        ePbrEmissive,
        ePbrEmissiveIntensity,

        ePbrUseAoMap,
        ePbrTexAoMap,

        ePbrMax
    } eStingrayPBSProperty;
    
    void Construct(FbxObject const *from) override;
    const char *GetTypeName() const override;
    FbxProperty GetProperty(int pId);

    FbxPropertyT<FbxInt> TypeId;

    FbxPropertyT<FbxDouble3> GlobalDiffuseCube;
    FbxPropertyT<FbxDouble3> GlobalSpecularCube;
    FbxPropertyT<FbxDouble3> BrdfLut;

    FbxPropertyT<FbxDouble2> UvOffset;
    FbxPropertyT<FbxDouble2> UvScale;

    FbxPropertyT<FbxFloat> UseNormalMap;
    FbxPropertyT<FbxDouble3> NormalMap;

    FbxPropertyT<FbxFloat> UseColorMap;
    FbxPropertyT<FbxDouble3> ColorMap;
    FbxPropertyT<FbxDouble3> BaseColor;

    FbxPropertyT<FbxFloat> UseMetallicMap;
    FbxPropertyT<FbxDouble3> MetallicMap;
    FbxPropertyT<FbxFloat> Metallic;

    FbxPropertyT<FbxFloat> UseRoughnessMap;
    FbxPropertyT<FbxDouble3> RoughnessMap;
    FbxPropertyT<FbxFloat> Roughness;

    FbxPropertyT<FbxFloat> UseEmissiveMap;
    FbxPropertyT<FbxDouble3> EmissiveMap;
    FbxPropertyT<FbxDouble3> Emissive;
    FbxPropertyT<FbxFloat> EmissiveIntensity;

    FbxPropertyT<FbxFloat> UseAoMap;
    FbxPropertyT<FbxDouble3> AoMap;

protected:
    void Destruct(bool pRecursive) override;
    void ConstructProperties(bool pForceSet) override;
};

