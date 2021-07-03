// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"

#include <filesystem>
#include <optional>

#include "fbx_standard_surface.h"
#include <fbxsdk.h>
#include <mesh_export.h>
#include <stb_image_write.h>
#include <glm/geometric.hpp>

#include <uuid.h>

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

#define UV_ELEMENT_NAME "default"

using index_t = std::make_signed<size_t>::type;

class ITexture {
public:
    virtual ~ITexture() = default;

    virtual void const *buffer() const = 0;
    virtual int width() const = 0;
    virtual int height() const = 0;
};

class IMesh {
public:
    virtual ~IMesh() = default;

    virtual int
    numElements() const = 0;
    virtual int
    element(int idx) const = 0;

    virtual int
    numPositions() const = 0;
    virtual FbxVector4 position(int idx) const = 0;

    virtual int
    numNormals() const = 0;
    virtual FbxVector4 normal(int idx) const = 0;

    virtual int
    numUVs() const = 0;
    virtual FbxVector2 uv(int idx) const = 0;
};

struct Material {
    ITexture const *base;
    ITexture const *normal;
    ITexture const *height;
    ITexture const *roughness;
    ITexture const *ao;
};

struct FBX_Context {
    FbxManager *sdkManager = nullptr;
    FbxIOSettings *ioSettings = nullptr;
    FbxScene *scene = nullptr;
    /** Implementation of the Arnold surface */
    FbxImplementation *implementation = nullptr;

    void
    Destroy() {
        sdkManager->Destroy();
        memset(this, 0, sizeof(*this));
    }
};


static bool write_texture(ITexture const *tex, std::string const &name, std::string &path) {
    auto tmpdir_path = std::filesystem::temp_directory_path();
    auto filename = std::filesystem::path("trigen_" + std::string(name) + ".png");
    path = (tmpdir_path / filename).string();

    if (stbi_write_png(path.c_str(), tex->width(), tex->height(), 3, tex->buffer(), tex->width() * 3) > 0) {
        return true;
    }

    fprintf(stderr, "failed to write texture '%s' to file '%s'\n", name.c_str(), path.c_str());

    return false;
}

static FbxFileTexture *create_texture(FbxScene *container, ITexture const *tex, std::string const &uuid, char const *kind) {
    auto const name = std::string(kind) + '.' + uuid;
    std::string path;

    if (!write_texture(tex, name, path)) {
        return nullptr;
    }

    auto texture = FbxFileTexture::Create(container, name.c_str());
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

static FbxFileTexture *create_texture(FbxScene *container, ITexture const *tex, std::string const &uuid, char const *kind, FbxProperty &tex_prop) {
    auto texture = create_texture(container, tex, uuid, kind);
    tex_prop.ConnectSrcObject(texture);
    return texture;
}

static bool get_winding_order(IMesh const *mesh, int i0, int i1, int i2) {
    auto faceNormal = mesh->normal(i0) + mesh->normal(i1) + mesh->normal(i2);
    faceNormal.Normalize();

    auto v0 = mesh->position(i2) - mesh->position(i0);
    auto v1 = mesh->position(i2) - mesh->position(i1);
    auto c = v0.CrossProduct(v1);
    c.Normalize();

    return abs(1 - (c.DotProduct(faceNormal))) < 0.001f;
}

using Build_Mesh_Callback = std::function<void(std::string const &uuid, FbxNode *node, FbxMesh *mesh)>;

static void build_mesh(FBX_Context &ctx, IMesh const *mesh, Build_Mesh_Callback const &then) {
    auto const uuid = uuids::to_string(uuids::uuid_system_generator {}());

    auto const meshName = uuid + ".mesh";
    auto meshNode = FbxMesh::Create(ctx.scene, meshName.c_str());

    auto num_control_points = mesh->numPositions();

    meshNode->InitControlPoints(num_control_points);

    // Fill control point array with vertex positions
    auto *controlPoints = meshNode->GetControlPoints();
    for (int i = 0; i < num_control_points; i++) {
        controlPoints[i] = mesh->position(i);
    }

    // Create normal array
    auto *normal_element = meshNode->CreateElementNormal();
    // eByControlPoint: there is a normal vector for each control point
    // i.e. if multiple polygons share a vertex, then each that vertex will
    // have the same normal vector
    normal_element->SetMappingMode(FbxGeometryElement::eByPolygonVertex);
    // eDirect: mapping for the nth control point is found in the nth place of
    // the direct array
    normal_element->SetReferenceMode(FbxGeometryElement::eIndexToDirect);

    // Create UV array
    auto *uv_element = meshNode->CreateElementUV(UV_ELEMENT_NAME);
    uv_element->SetMappingMode(FbxGeometryElement::eByPolygonVertex);
    // uv_element->SetReferenceMode(FbxGeometryElement::eDirect);
    uv_element->SetReferenceMode(FbxGeometryElement::eIndexToDirect);

    // Create inMaterial array
    auto *mat_element = meshNode->CreateElementMaterial();
    mat_element->SetMappingMode(FbxGeometryElement::eAllSame);
    mat_element->SetReferenceMode(FbxGeometryElement::eIndexToDirect);

    // Fill normal array
    auto &normal_array = normal_element->GetDirectArray();
    for (int i = 0; i < mesh->numNormals(); i++) {
        normal_array.Add(mesh->normal(i));
    }
    auto &normal_index_array = normal_element->GetIndexArray();
    for (int i = 0; i < mesh->numElements(); i++) {
        normal_index_array.Add(mesh->element(i));
    }

    // Fill UV array
    auto &uv_array = uv_element->GetDirectArray();
    for (int i = 0; i < mesh->numUVs(); i++) {
        uv_array.Add(mesh->uv(i));
    }
    auto &uv_index_array = uv_element->GetIndexArray();
    for (int i = 0; i < mesh->numElements(); i++) {
        uv_index_array.Add(mesh->element(i));
    }

    // Add triangles
    auto num_triangles = mesh->numElements() / 3;
    for (size_t t = 0; t < num_triangles; t++) {
        auto i0 = mesh->element(t * 3 + 0);
        auto i1 = mesh->element(t * 3 + 1);
        auto i2 = mesh->element(t * 3 + 2);

        meshNode->BeginPolygon(-1, -1, -1, false);
        if (get_winding_order(mesh, i0, i1, i2)) {
            meshNode->AddPolygon(i1, i1);
            meshNode->AddPolygon(i0, i0);
        } else {
            meshNode->AddPolygon(i0, i0);
            meshNode->AddPolygon(i1, i1);
        }
        meshNode->AddPolygon(i2, i2);
        meshNode->EndPolygon();
    }
    meshNode->BuildMeshEdgeArray();

    meshNode->GenerateTangentsData(UV_ELEMENT_NAME);

    auto const nodeName = uuid + ".node";
    auto node = FbxNode::Create(ctx.scene, nodeName.c_str());
    node->SetNodeAttribute(meshNode);
    node->SetShadingMode(FbxNode::eTextureShading);

    auto rootNode = ctx.scene->GetRootNode();
    rootNode->AddChild(node);

    then(uuid, node, meshNode);
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

static bool save_scene(FBX_Context &ctx, char const *path) {
    bool status = false;

    auto exporter = configure_exporter(ctx.sdkManager, ctx.ioSettings, path);

    if (exporter != nullptr) {
        status = exporter->Export(ctx.scene);
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

static std::optional<FBX_Context>
make_context() {
    FbxManager *sdkManager;
    FbxIOSettings *ioSettings;
    FbxScene *scene;
    FbxImplementation *implementation;
    FbxBindingTable *bindingTable;
    bool ret = false;

    if (!create_sdk_objects(&sdkManager, &ioSettings)) {
        return std::nullopt;
    }

    scene = FbxScene::Create(sdkManager, "scene");

    if (scene == nullptr) {
        goto err_scene;
    }

    implementation = fbx_standard_surface_create_implementation(scene);
    if (implementation == nullptr) {
        goto err_implementation;
    }

    bindingTable = fbx_standard_surface_create_binding_table(implementation);
    if (bindingTable == nullptr) {
        goto err_bindingTable;
    }

    return FBX_Context { sdkManager, ioSettings, scene, implementation };

err_bindingTable:
    implementation->Destroy();
err_implementation:
    scene->Destroy();
err_scene:
    ioSettings->Destroy();
    sdkManager->Destroy();
err_ret:
    return std::nullopt;
}

class TrigenMesh : public IMesh {
public:
    TrigenMesh(Trigen_Mesh const *mesh)
        : _mesh(mesh) {
    }

    int
    numElements() const override {
        return _mesh->triangle_count * 3;
    }

    int
    element(int idx) const override {
        return _mesh->indices[idx];
    }

    int
    numPositions() const override {
        return _mesh->position_count;
    }

    FbxVector4
    position(int idx) const override {
        auto *vec = &_mesh->positions[idx * 3];
        return FbxVector4(vec[0], vec[1], vec[2]);
    }
    int
    numNormals() const override {
        return _mesh->normal_count;
    }
    FbxVector4
    normal(int idx) const override {
        auto *vec = &_mesh->normals[idx * 3];
        return FbxVector4(vec[0], vec[1], vec[2]);
    }
    int
    numUVs() const override {
        return _mesh->position_count;
    }
    FbxVector2
    uv(int idx) const override {
        auto *vec = &_mesh->uvs[idx * 2];
        return FbxVector2(vec[0], vec[1]);
    }

private:
    Trigen_Mesh const *_mesh;
};

class TrigenTexture : public ITexture {
public:
    TrigenTexture(Trigen_Texture const *tex)
        : _tex(tex) {
    }
    ~TrigenTexture() override = default;

    void const *buffer() const override {
        return _tex->image;
    }

    int width() const override {
        return _tex->width;
    }

    int height() const override {
        return _tex->height;
    }

private:
    Trigen_Texture const *_tex;
};

static void
add_tree_mesh(FBX_Context &ctx, Export_Model const &model) {
    auto mesh = TrigenMesh(&model.mesh);
    TrigenTexture texBase(model.material.base);
    TrigenTexture texNormal(model.material.normal);
    TrigenTexture texHeight(model.material.height);
    TrigenTexture texRoughness(model.material.roughness);
    TrigenTexture texAo(model.material.ao);

    Material material = {
        &texBase,
        &texNormal,
        &texHeight,
        &texRoughness,
        &texAo,
    };

    build_mesh(ctx, &mesh, [&](std::string const &uuid, FbxNode *node, FbxMesh *mesh) {
        auto surfName = uuid + ".surf";
        auto surf = FbxArnoldStandardSurface(ctx.scene, surfName.c_str());
        surf.material()->AddImplementation(ctx.implementation);
        surf.material()->SetDefaultImplementation(ctx.implementation);

        create_texture(ctx.scene, material.base, uuid, "base_color", surf.get_baseColor());
        create_texture(ctx.scene, material.normal, uuid, "normal", surf.get_normalCamera());
        create_texture(ctx.scene, material.roughness, uuid, "roughness", surf.get_specularRoughness());
        create_texture(ctx.scene, material.ao, uuid, "ao");
        create_texture(ctx.scene, material.height, uuid, "height");

        node->AddMaterial(surf.material());
        mesh->GetElementMaterial()->GetIndexArray().Add(0);
    });
}

static void
add_foliage_mesh(FBX_Context& ctx, Export_Model const& model) {
    auto mesh = TrigenMesh(&model.foliageMesh);
    TrigenTexture texDiffuse(model.leafTexture);

    build_mesh(ctx, &mesh, [&](std::string const &uuid, FbxNode *node, FbxMesh *mesh) {
        auto surfName = uuid + ".surf";
        auto surf = FbxSurfacePhong::Create(ctx.scene, surfName.c_str());

        create_texture(ctx.scene, &texDiffuse, uuid, "base_color", surf->Diffuse);

        node->AddMaterial(surf);
        mesh->GetElementMaterial()->GetIndexArray().Add(0);
    });
}

bool
fbx_try_save(char const *path, Export_Model const &model) {
    auto maybeCtx = make_context();
    if (!maybeCtx) {
        return false;
    }

    add_tree_mesh(*maybeCtx, model);
    add_foliage_mesh(*maybeCtx, model);
    auto ret = save_scene(*maybeCtx, path);

    maybeCtx->Destroy();

    return ret;
}
