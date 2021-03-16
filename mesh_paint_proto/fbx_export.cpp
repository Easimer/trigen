#include "fbx_export.h"

#include <fbxsdk.h>

static void build_mesh(FbxScene *scene, PSP::Mesh const *mesh, PSP::Material const *material) {
    auto meshNode = FbxMesh::Create(scene, "plant mesh");

    auto num_control_points_zu = mesh->position.size();
    assert(num_control_points_zu < INT_MAX);
    auto num_control_points = int(num_control_points_zu);

    meshNode->InitControlPoints(num_control_points);

    auto controlPoints = meshNode->GetControlPoints();
    for (int i = 0; i < num_control_points; i++) {
        auto &pos = mesh->position[i];
        controlPoints[i] = FbxVector4(pos.x, pos.y, pos.z);
    }

    auto num_triangles = mesh->elements.size() / 3;
    for (size_t t = 0; t < num_triangles; t++) {
        auto i0 = mesh->elements[t * 3 + 0];
        auto i1 = mesh->elements[t * 3 + 1];
        auto i2 = mesh->elements[t * 3 + 2];

        meshNode->BeginPolygon();
        meshNode->AddPolygon(i0);
        meshNode->AddPolygon(i1);
        meshNode->AddPolygon(i2);
        meshNode->EndPolygon();
    }

    auto node = FbxNode::Create(scene, "plant");
    node->SetNodeAttribute(meshNode);

    auto rootNode = scene->GetRootNode();
    rootNode->AddChild(node);
}

static bool save_scene(FbxManager *manager, FbxDocument *scene, char const *path) {
    bool status;
    auto exporter = FbxExporter::Create(manager, "");

    // Write in fall back format in less no ASCII format found
    auto file_format = manager->GetIOPluginRegistry()->GetNativeWriterFormat();

    //Try to export in binary if possible
    auto num_formats = manager->GetIOPluginRegistry()->GetWriterFormatCount();

    for (int format_idx = 0; format_idx < num_formats; format_idx++) {
        if (manager->GetIOPluginRegistry()->WriterIsFBX(format_idx)) {
            FbxString desc = manager->GetIOPluginRegistry()->GetWriterFormatDescription(format_idx);
            if (desc.Find("binary") >= 0) {
                file_format = format_idx;
                break;
            }
        }
    }

    if(exporter->Initialize(path, file_format, manager->GetIOSettings()) == false) {
        FBXSDK_printf("Call to FbxExporter::Initialize() failed.\n");
        FBXSDK_printf("Error returned: %s\n\n", exporter->GetStatus().GetErrorString());
        status = false;
        goto err_exporter;
    }

    status = exporter->Export(scene); 

err_exporter:
    exporter->Destroy();

    return status;
}

bool fbx_try_save(char const *path, PSP::Mesh const *mesh, PSP::Material const *material) {
    FbxManager *sdkManager;
    FbxIOSettings *ioSettings;
    FbxExporter *exporter;
    FbxScene *scene;
    bool ret = false;

    sdkManager = FbxManager::Create();
    ioSettings = FbxIOSettings::Create(sdkManager, IOSROOT);
    scene = FbxScene::Create(sdkManager, "scene");

    if (scene == NULL) {
        goto err_manager;
    }

    build_mesh(scene, mesh, material);
    save_scene(sdkManager, scene, path);

err_scene:
    scene->Destroy();
err_manager:
    ioSettings->Destroy();
    sdkManager->Destroy();
    return ret;
}
