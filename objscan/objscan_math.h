// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#pragma once

#include <memory>
#include <vector>
#include <glm/glm.hpp>
#include <objscan.h>
#include "tiny_obj_loader.h"
#include "mesh.h"


// Determines whether the segment `origin -> origin + dir` intersects any triangles or not.
bool intersects_any(
                    glm::vec3 origin, glm::vec3 dir,
                    Mesh* mesh
                    );

std::vector<glm::vec4> sample_points(
                                     float x_min, float x_max, float x_step,
                                     float y_min, float y_max, float y_step,
                                     float z_min, float z_max, float z_step,
                                     Mesh* mesh
                                     );

std::vector<std::pair<int, int>> form_connections(
                                                  int offset, int count,
                                                  objscan_position const* positions, int N,
                                                  float step_x, float step_y, float step_z,
                                                  Mesh* mesh
                                                  );

void calculate_bounding_box(
                            float& x_min, float& y_min, float& z_min,
                            float& x_max, float& y_max, float& z_max,
                            tinyobj::attrib_t const& attrib);