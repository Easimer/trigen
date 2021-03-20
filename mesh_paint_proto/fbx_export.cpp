// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "fbx_export.h"
#include "fbx_standard_surface.h"
#include <fbxsdk.h>
#include <filesystem>
#include <stb_image_write.h>

// Forces exporter to use Wavefront OBJ as output format
// #define DEBUG_EXPORT_OBJ (0)
// Forces exporter to output ASCII FBX
// #define DEBUG_EXPORT_ASCII_FBX (0)

#if DEBUG_EXPORT_OBJ
#define FORMAT_FILTER_STRING "obj"
#define DONT_FILTER_NONFBX (true)
#else
#if DEBUG_EXPORT_ASCII_FBX
#define FORMAT_FILTER_STRING "ascii"
#else
#define FORMAT_FILTER_STRING "binary"
#endif
#define DONT_FILTER_NONFBX (false)
#endif

#define UV_ELEMENT_NAME "uv"

static bool write_texture(PSP::Texture const &tex, char const *kind, std::string &path) {
    auto tmpdir_path = std::filesystem::temp_directory_path();
    auto filename = std::filesystem::path("trigen_" + std::string(kind) + ".png");
    path = (tmpdir_path / filename).string();

    if (stbi_write_png(path.c_str(), tex.width, tex.height, 3, tex.buffer, tex.width * 3) > 0) {
        return true;
    }

    fprintf(stderr, "failed to write texture '%s' to file '%s'\n", kind, path.c_str());

    return false;
}

static FbxFileTexture *create_texture(FbxScene *container, PSP::Texture const &tex, char const *kind) {
    std::string path;
    if (!write_texture(tex, kind, path)) {
        return nullptr;
    }

    auto texture = FbxFileTexture::Create(container, kind);
    texture->SetFileName(path.c_str());
    texture->SetTextureUse(FbxTexture::eStandard);
    texture->SetMappingType(FbxTexture::eUV);
    texture->SetMaterialUse(FbxFileTexture::eModelMaterial);
    texture->SetSwapUV(false);
    texture->SetTranslation(0.0, 0.0);
    texture->SetScale(1.0, 1.0);
    texture->SetRotation(0.0, 0.0);
    texture->UVSet.Set(FbxString(UV_ELEMENT_NAME));

    return texture;
}

static FbxFileTexture *create_texture(FbxScene *container, PSP::Texture const &tex, char const *kind, FbxProperty &tex_prop) {
    auto texture = create_texture(container, tex, kind);
    tex_prop.ConnectSrcObject(texture);
    return texture;
}

static void build_mesh(FbxScene *scene, PSP::Mesh const *mesh, PSP::Material const *material) {
    auto meshNode = FbxMesh::Create(scene, "mesh");

    auto num_control_points_zu = mesh->position.size();
    assert(num_control_points_zu < INT_MAX);
    auto num_control_points = int(num_control_points_zu);

    meshNode->InitControlPoints(num_control_points);

    // Fill control point array with vertex positions
    auto controlPoints = meshNode->GetControlPoints();
    for (int i = 0; i < num_control_points; i++) {
        auto &pos = mesh->position[i];
        controlPoints[i] = FbxVector4(pos.x, pos.y, pos.z);
    }

    // Create normal array
    auto normal_element = meshNode->CreateElementNormal();
    normal_element->SetMappingMode(FbxGeometryElement::eByControlPoint);
    normal_element->SetReferenceMode(FbxGeometryElement::eDirect);

    // Create UV array
    auto uv_element = meshNode->CreateElementUV(UV_ELEMENT_NAME);
    uv_element->SetMappingMode(FbxGeometryElement::eByPolygonVertex);
    // uv_element->SetReferenceMode(FbxGeometryElement::eDirect);
    uv_element->SetReferenceMode(FbxGeometryElement::eIndexToDirect);

    // Create vertex color array
    auto vtxcol_element = meshNode->CreateElementVertexColor();
    vtxcol_element->SetMappingMode(FbxGeometryElement::eByControlPoint);
    vtxcol_element->SetReferenceMode(FbxGeometryElement::eDirect);

    // Create material array
    auto mat_element = meshNode->CreateElementMaterial();
    mat_element->SetMappingMode(FbxGeometryElement::eAllSame);
    mat_element->SetReferenceMode(FbxGeometryElement::eIndexToDirect);

    // Fill normal and vertex color array
    auto &normal_array = normal_element->GetDirectArray();
    auto &vtxcol_array = vtxcol_element->GetDirectArray();
    for (int i = 0; i < num_control_points; i++) {
        auto &normal = mesh->normal[i];
        auto &col = mesh->chart_debug_color[i];
        normal_array.Add(FbxVector4(normal.x, normal.y, normal.z));
        vtxcol_array.Add(FbxColor(float(col.r) / 255.f, float(col.g) / 255.f, float(col.b) / 255.f));
    }

    // Fill UV array
    auto &uv_array = uv_element->GetDirectArray();
    auto num_elements = mesh->uv.size();
    for (int i = 0; i < num_elements; i++) {
        auto &uv = mesh->uv[i];
        uv_array.Add(FbxVector2(uv.x, uv.y));
    }

    // Add triangles
    auto num_triangles = mesh->elements.size() / 3;
    for (size_t t = 0; t < num_triangles; t++) {
        auto off0 = t * 3 + 0;
        auto off1 = t * 3 + 1;
        auto off2 = t * 3 + 2;
        auto i0 = mesh->elements[off0];
        auto i1 = mesh->elements[off1];
        auto i2 = mesh->elements[off2];

        meshNode->BeginPolygon();
        meshNode->AddPolygon(i0, off0);
        meshNode->AddPolygon(i1, off1);
        meshNode->AddPolygon(i2, off2);
        meshNode->EndPolygon();
    }
    meshNode->BuildMeshEdgeArray();

    auto node = FbxNode::Create(scene, "plant");
    node->SetNodeAttribute(meshNode);

    auto implementation = fbx_standard_surface_create_implementation(scene);
    auto bindingTable = fbx_standard_surface_create_binding_table(implementation);

    auto surf = FbxArnoldStandardSurface(scene, "plant_material");
    surf.material()->AddImplementation(implementation);
    surf.material()->SetDefaultImplementation(implementation);

    create_texture(scene, material->base, "base_color", surf.get_baseColor());
    create_texture(scene, material->normal, "normal", surf.get_normalCamera());
    create_texture(scene, material->roughness, "roughness", surf.get_diffuseRoughness());
    create_texture(scene, material->ao, "ao");
    create_texture(scene, material->height, "height");

    node->AddMaterial(surf.material());

    // Fill material array
    mat_element->GetIndexArray().Add(0);

    auto rootNode = scene->GetRootNode();
    rootNode->AddChild(node);
}

static FbxExporter *configure_exporter(FbxManager *manager, FbxIOSettings *ios, char const *path) {
    FbxExporter *exporter;
    int num_formats;
    int file_format;

    exporter = FbxExporter::Create(manager, "");
    if (!exporter) {
        return nullptr;
    }

    // Choose native format as a fallback
    file_format = manager->GetIOPluginRegistry()->GetNativeWriterFormat();

    // Enumerate output formats
    num_formats = manager->GetIOPluginRegistry()->GetWriterFormatCount();
    for (int format_idx = 0; format_idx < num_formats; format_idx++) {
        if (DONT_FILTER_NONFBX || manager->GetIOPluginRegistry()->WriterIsFBX(format_idx)) {
            FbxString desc = manager->GetIOPluginRegistry()->GetWriterFormatDescription(format_idx);
            // Prefer binary FBX
            if (desc.Find(FORMAT_FILTER_STRING) >= 0) {
                file_format = format_idx;
                break;
            }
        }
    }

    if (true || manager->GetIOPluginRegistry()->WriterIsFBX(file_format)) {
        // If we're using FBX then embed the textures
        manager->GetIOSettings()->SetBoolProp(EXP_FBX_TEXTURE, true);
        manager->GetIOSettings()->SetBoolProp(EXP_FBX_MATERIAL, true);
        manager->GetIOSettings()->SetBoolProp(EXP_FBX_EMBEDDED, true);
    }

    // Initialize exporter
    if(!exporter->Initialize(path, file_format, manager->GetIOSettings())) {
        FBXSDK_printf("Call to FbxExporter::Initialize() failed.\n");
        FBXSDK_printf("Error returned: %s\n\n", exporter->GetStatus().GetErrorString());
        exporter->Destroy();
        return nullptr;
    }

    return exporter;
}

static bool save_scene(FbxManager *manager, FbxIOSettings *ios, FbxDocument *scene, char const *path) {
    bool status = false;

    auto exporter = configure_exporter(manager, ios, path);

    if (exporter != nullptr) {
        status = exporter->Export(scene);
        exporter->Destroy();
    }

    return status;
}

static bool create_sdk_objects(FbxManager **out_sdkManager, FbxIOSettings **out_ioSettings) {
    FbxManager *sdkManager;
    FbxIOSettings *ioSettings;

    assert(out_sdkManager != nullptr && out_ioSettings != nullptr);

    sdkManager = FbxManager::Create();
    if (sdkManager == nullptr) {
        return false;
    }

    ioSettings = FbxIOSettings::Create(sdkManager, IOSROOT);
    if (ioSettings == nullptr) {
        sdkManager->Destroy();
        return false;
    }

    sdkManager->SetIOSettings(ioSettings);

    *out_sdkManager = sdkManager;
    *out_ioSettings = ioSettings;
    return true;
}

bool fbx_try_save(char const *path, PSP::Mesh const *mesh, PSP::Material const *material) {
    FbxManager *sdkManager;
    FbxIOSettings *ioSettings;
    FbxScene *scene;
    bool ret = false;

    if (!create_sdk_objects(&sdkManager, &ioSettings)) {
        goto err_ret;
    }

    scene = FbxScene::Create(sdkManager, "scene");

    if (scene == NULL) {
        goto err_sdk;
    }

    build_mesh(scene, mesh, material);
    save_scene(sdkManager, ioSettings, scene, path);

    scene->Destroy();
err_sdk:
    ioSettings->Destroy();
    sdkManager->Destroy();
err_ret:
    return ret;
}
