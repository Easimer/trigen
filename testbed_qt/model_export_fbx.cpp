// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: exports models into an FBX file
//

#include "common.h"
#include "model_export.h"
#include <cassert>

#include <fbxsdk.h>

class FBX_Exporter : public IModel_Exporter {
public:
	FBX_Exporter() {
		sdkManager = FbxManager::Create();
		assert(sdkManager != NULL);

		ioSettings = FbxIOSettings::Create(sdkManager, IOSROOT);
		assert(ioSettings != NULL);

		exporter = FbxExporter::Create(sdkManager, "");
		assert(exporter != NULL);
	}

	~FBX_Exporter() override {
		if (exporter != NULL) {
			exporter->Destroy();
		}

		if (ioSettings != NULL) {
			ioSettings->Destroy();
		}

		if (sdkManager != NULL) {
			sdkManager->Destroy();
		}
	}
private:
	void Export(char const* path, Optimized_Mesh<TG_Vertex> const& model) override {
		throw std::runtime_error("Not Implemented Yet");
	}

private:
	FbxManager* sdkManager;
	FbxIOSettings* ioSettings;
	FbxExporter* exporter;
};

std::unique_ptr<IModel_Exporter> create_fbx_model_exporter() {
	return std::make_unique<FBX_Exporter>();
}