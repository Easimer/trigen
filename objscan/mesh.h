// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#pragma once

#include <vector>
#include <glm/vec3.hpp>
#include "tiny_obj_loader.h"

struct Triangle {
	glm::vec3 v0, v1, v2;
};

struct Bounding_Box {
	glm::vec3 min, max;
};

struct Mesh {
	std::vector<Bounding_Box> bounding_boxes;
	std::vector<Triangle> triangles;
};

inline Mesh create_mesh(std::vector<tinyobj::shape_t> const& shapes, tinyobj::attrib_t const& attrib) {
	std::vector<Triangle> triangles;
	std::vector<Bounding_Box> bounding_boxes;
	Triangle t;
	Bounding_Box bb;

	for (auto& shape : shapes) {
		auto& indices = shape.mesh.indices;
		auto N = indices.size() / 3;
		
		for (int i = 0; i < N; i++) {
			auto i0 = indices[i * 3 + 0].vertex_index;
			auto i1 = indices[i * 3 + 1].vertex_index;
			auto i2 = indices[i * 3 + 2].vertex_index;
			
			t.v0 = *(glm::vec3*)&attrib.vertices[i0 * 3];
			t.v1 = *(glm::vec3*)&attrib.vertices[i1 * 3];
			t.v2 = *(glm::vec3*)&attrib.vertices[i2 * 3];

			triangles.push_back(t);

			bb.min = { INFINITY, INFINITY, INFINITY };
			bb.max = { -INFINITY, -INFINITY, -INFINITY };

			for (int i = 0; i < 3; i++) {
				if (t.v0[i] < bb.min[i]) bb.min[i] = t.v0[i];
				if (t.v1[i] < bb.min[i]) bb.min[i] = t.v1[i];
				if (t.v2[i] < bb.min[i]) bb.min[i] = t.v2[i];

				if (t.v0[i] > bb.max[i]) bb.max[i] = t.v0[i];
				if (t.v1[i] > bb.max[i]) bb.max[i] = t.v1[i];
				if (t.v2[i] > bb.max[i]) bb.max[i] = t.v2[i];

				// Prevent zero-size bounding boxes
				if (bb.max[i] - bb.min[i] == 0.0f) {
					bb.max[i] = bb.min[i] + 0.1f;
				}
			}

			bounding_boxes.push_back(bb);
		}
	}

	return { bounding_boxes, triangles };
}