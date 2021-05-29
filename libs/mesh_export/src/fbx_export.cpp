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

    virtual int numControlPoints() const = 0;
    virtual FbxVector4 position(int idxControlPoint) const = 0;

    virtual int numTriangles() const = 0;
    virtual std::tuple<int, int, int> vertexIndices(int idxTriangle) const = 0;
    virtual std::tuple<int, int, int> uvIndices(int idxTriangle) const = 0;

    virtual FbxVector4 normal(int idxControlPoint) const = 0;
    virtual FbxColor vertexColor(int idx) const = 0;

    virtual int numUVs() const = 0;
    virtual glm::vec2 uv(int idx) const = 0;
};

struct Material {
    ITexture const *base;
    ITexture const *normal;
    ITexture const *height;
    ITexture const *roughness;
    ITexture const *ao;
};

class PSPTexture : public ITexture {
public:
    PSPTexture(PSP::Texture const *tex)
        : _tex(tex) { }

    ~PSPTexture() override = default;

    void const *buffer() const override {
        return _tex->buffer;
    }

    int width() const override {
        return _tex->width;
    }

    int height() const override {
        return _tex->height;
    }

private:
    PSP::Texture const *_tex;
};

class PSPMesh : public IMesh {
public:
    PSPMesh(PSP::Mesh const *mesh) : _mesh(mesh) {}
    ~PSPMesh() override = default;

    int numControlPoints() const override {
        size_t ret_zu = _mesh->position.size();
        assert(ret_zu < INT_MAX);
        return int(ret_zu);
    }

    FbxVector4 position(int idxControlPoint) const override {
        auto &pos = _mesh->position[idxControlPoint];
        return FbxVector4(pos.x, pos.y, pos.z);
    }

    int numTriangles() const override {
        return _mesh->elements.size() / 3;
    }

    std::tuple<int, int, int> vertexIndices(int idxTriangle) const override {
        auto [off0, off1, off2] = uvIndices(idxTriangle);
        return { _mesh->elements[off0], _mesh->elements[off1], _mesh->elements[off2] };
    }

    std::tuple<int, int, int> uvIndices(int idxTriangle) const override {
        return { idxTriangle * 3 + 0,  idxTriangle * 3 + 1, idxTriangle * 3 + 2 };
    }

    FbxVector4 normal(int idxControlPoint) const override {
        auto &normal = _mesh->normal[idxControlPoint];
        return FbxVector4(normal.x, normal.y, normal.z);
    }

    FbxColor vertexColor(int idx) const override {
        auto &col = _mesh->chart_debug_color[idx];
        return FbxColor(float(col.r) / 255.f, float(col.g) / 255.f, float(col.b) / 255.f);
    }

    int numUVs() const override {
        return _mesh->uv.size();
    }

    glm::vec2 uv(int idx) const override {
        return _mesh->uv[idx];
    }

private:
    PSP::Mesh const *_mesh;
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

template<typename Vec>
static void compress_mesh_attribute(std::vector<Vec> &compressedDirectArray, std::vector<int> &indexArray, std::vector<Vec> const &directArray) {
    float const epsilon = 0.01f;

    for (index_t i = 0; i < directArray.size(); i++) {
        std::optional<index_t> idx;
        auto current = directArray[i];

        for (index_t j = compressedDirectArray.size() - 1; j >= 0; j--) {
            if (distance(compressedDirectArray[j], current) < epsilon) {
                idx = j;
                break;
            }
        }

        if (!idx.has_value()) {
            idx = compressedDirectArray.size();
            compressedDirectArray.push_back(current);
        }

        assert(idx.has_value());

        indexArray.push_back(idx.value());
    }

    assert(indexArray.size() == directArray.size());
}

static void make_texcoords_attribute_array_indirect(IMesh const *mesh, FbxLayerElementUV *uv_element) {
    std::vector<glm::vec2> uvDirectArray;

    std::vector<glm::vec2> compressedUVDirectArray;
    std::vector<int> uvIndexArray;

    // Put the texcoord data into a vector
    auto num_elements = mesh->numUVs();
    uvDirectArray.reserve(num_elements);

    for (int i = 0; i < num_elements; i++) {
        uvDirectArray.push_back(mesh->uv(i));
    }

    // Convert our texcoord data into an indirect format
    compress_mesh_attribute(compressedUVDirectArray, uvIndexArray, uvDirectArray);

    // Put the results into the layer element
    auto &uvDirectArrayFbx = uv_element->GetDirectArray();
    auto &uvIndexArrayFbx = uv_element->GetIndexArray();

    for (auto &vec : compressedUVDirectArray) {
        uvDirectArrayFbx.Add(FbxVector2(vec.x, vec.y));
    }
    for (auto idx : uvIndexArray) {
        assert(idx < uvDirectArrayFbx.GetCount());
        uvIndexArrayFbx.Add(idx);
    }
}

static void build_mesh(FbxScene *scene, IMesh const *mesh, Material const &material) {
    auto const uuid = uuids::to_string(uuids::uuid_system_generator {}());

    auto const meshName = uuid + ".mesh";
    auto meshNode = FbxMesh::Create(scene, meshName.c_str());

    auto num_control_points = mesh->numControlPoints();

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
    normal_element->SetMappingMode(FbxGeometryElement::eByControlPoint);
    // eDirect: mapping for the nth control point is found in the nth place of
    // the direct array
    normal_element->SetReferenceMode(FbxGeometryElement::eDirect);

    // Create UV array
    auto *uv_element = meshNode->CreateElementUV(UV_ELEMENT_NAME);
    uv_element->SetMappingMode(FbxGeometryElement::eByPolygonVertex);
    // uv_element->SetReferenceMode(FbxGeometryElement::eDirect);
    uv_element->SetReferenceMode(FbxGeometryElement::eIndexToDirect);

    // Create vertex color array
    auto *vtxcol_element = meshNode->CreateElementVertexColor();
    vtxcol_element->SetMappingMode(FbxGeometryElement::eByControlPoint);
    vtxcol_element->SetReferenceMode(FbxGeometryElement::eDirect);

    // Create inMaterial array
    auto *mat_element = meshNode->CreateElementMaterial();
    mat_element->SetMappingMode(FbxGeometryElement::eAllSame);
    mat_element->SetReferenceMode(FbxGeometryElement::eIndexToDirect);

    // Fill normal and vertex color array
    auto &normal_array = normal_element->GetDirectArray();
    auto &vtxcol_array = vtxcol_element->GetDirectArray();
    for (int i = 0; i < num_control_points; i++) {
        normal_array.Add(mesh->normal(i));
        vtxcol_array.Add(mesh->vertexColor(i));
    }

    make_texcoords_attribute_array_indirect(mesh, uv_element);

    // Add triangles
    auto num_triangles = mesh->numTriangles();
    for (size_t t = 0; t < num_triangles; t++) {
        auto [i0, i1, i2] = mesh->vertexIndices(t);

        meshNode->BeginPolygon(-1, -1, -1, false);
        if (get_winding_order(mesh, i0, i1, i2)) {
            meshNode->AddPolygon(i1);
            meshNode->AddPolygon(i0);
        } else {
            meshNode->AddPolygon(i0);
            meshNode->AddPolygon(i1);
        }
        meshNode->AddPolygon(i2);
        meshNode->EndPolygon();
    }
    meshNode->BuildMeshEdgeArray();

    meshNode->GenerateTangentsData(UV_ELEMENT_NAME);

    auto const nodeName = uuid + ".node";
    auto node = FbxNode::Create(scene, nodeName.c_str());
    node->SetNodeAttribute(meshNode);
    node->SetShadingMode(FbxNode::eTextureShading);

    auto implementation = fbx_standard_surface_create_implementation(scene);
    auto bindingTable = fbx_standard_surface_create_binding_table(implementation);

    auto surfName = uuid + ".surf";
    auto surf = FbxArnoldStandardSurface(scene, surfName.c_str());
    surf.material()->AddImplementation(implementation);
    surf.material()->SetDefaultImplementation(implementation);

    create_texture(scene, material.base, uuid, "base_color", surf.get_baseColor());
    create_texture(scene, material.normal, uuid, "normal", surf.get_normalCamera());
    create_texture(scene, material.roughness, uuid, "roughness", surf.get_specularRoughness());
    create_texture(scene, material.ao, uuid, "ao");
    create_texture(scene, material.height, uuid, "height");

    node->AddMaterial(surf.material());

    // Fill inMaterial array
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

static bool fbx_try_save(char const *path, IMesh const *mesh, Material const &material) {
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
    ret = true;

    scene->Destroy();
err_sdk:
    ioSettings->Destroy();
    sdkManager->Destroy();
err_ret:
    return ret;
}

bool fbx_try_save(char const *path, PSP::Mesh const *inMesh, PSP::Material const *inMaterial) {
    PSPMesh mesh(inMesh);

    PSPTexture texBase(&inMaterial->base);
    PSPTexture texNormal(&inMaterial->normal);
    PSPTexture texHeight(&inMaterial->height);
    PSPTexture texRoughness(&inMaterial->roughness);
    PSPTexture texAo(&inMaterial->ao);

    Material material = {
        &texBase,
        &texNormal,
        &texHeight,
        &texRoughness,
        &texAo,
    };

    return fbx_try_save(path, &mesh, material);
}

class TrigenMesh : public IMesh {
public:
    TrigenMesh(Trigen_Mesh const *mesh)
        : _mesh(mesh) {
    }

    int numControlPoints() const override {
        return _mesh->position_count;
    }

    FbxVector4 position(int idxControlPoint) const override {
        auto pos = _mesh->positions + idxControlPoint * 3;
        return FbxVector4(pos[0], pos[1], pos[2]);
    }
    
    int numTriangles() const override {
        return _mesh->triangle_count;
    }

    std::tuple<int, int, int> vertexIndices(int idxTriangle) const override {
        auto [off0, off1, off2] = uvIndices(idxTriangle);
        return { _mesh->vertex_indices[off0], _mesh->vertex_indices[off1], _mesh->vertex_indices[off2] };
    }

    std::tuple<int, int, int> uvIndices(int idxTriangle) const override {
        return { idxTriangle * 3 + 0,  idxTriangle * 3 + 1, idxTriangle * 3 + 2 };
    }


    FbxVector4 normal(int idxControlPoint) const override {
        auto idx = _mesh->normal_indices[idxControlPoint];
        auto normal = _mesh->normals + idx * 3;
        return FbxVector4(normal[0], normal[1], normal[2]);
    }

    FbxColor vertexColor(int idx) const override {
        return FbxColor(0, 0, 0);
    }

    int numUVs() const override {
        return _mesh->triangle_count * 3;
    }

    glm::vec2 uv(int idx) const override {
        auto uv = _mesh->uvs + idx * 2;
        return glm::vec2(uv[0], uv[1]);
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

bool fbx_try_save(char const *path, Trigen_Mesh const &inMesh, Trigen_Material const &inMaterial) {
    TrigenMesh mesh(&inMesh);
    TrigenTexture texBase(inMaterial.base);
    TrigenTexture texNormal(inMaterial.normal);
    TrigenTexture texHeight(inMaterial.height);
    TrigenTexture texRoughness(inMaterial.roughness);
    TrigenTexture texAo(inMaterial.ao);

    Material material = {
        &texBase,
        &texNormal,
        &texHeight,
        &texRoughness,
        &texAo,
    };

    return fbx_try_save(path, &mesh, material);
}
