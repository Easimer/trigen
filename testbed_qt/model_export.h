// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: model export interface
//

#pragma once
#include <memory>
#include <trigen/meshbuilder.h>
#include <trigen/tree_meshifier.h>

class IModel_Exporter {
public:
	virtual ~IModel_Exporter() = default;
	virtual void Export(char const* path, Optimized_Mesh<TG_Vertex> const& model) = 0;
};

enum class Model_Format {
	FBX,
};

std::unique_ptr<IModel_Exporter> create_fbx_model_exporter();

inline std::unique_ptr<IModel_Exporter> create_model_exporter(Model_Format format) {
	switch (format) {
	case Model_Format::FBX:
		return create_fbx_model_exporter();
	}
}