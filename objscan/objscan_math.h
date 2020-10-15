// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#pragma once

#include <vector>
#include <glm/glm.hpp>
#include <objscan.h>
#include "tiny_obj_loader.h"

// Ray-triangle intersection
// Moeller-Trumbore algorithm
// Returns true on intersection and fills in `xp` and `t`, such that
// `origin + t * dir = xp`.
// Returns false otherwise.
bool ray_triangle_intersect(
                            glm::vec3& xp, float t,
                            glm::vec3 const& origin, glm::vec3 const& dir,
                            glm::vec3 const& v0, glm::vec3 const& v1, glm::vec3 const& v2
                            );

// Counts how many triangles does the given ray intersect.
int count_triangles_intersected(
                                glm::vec3 origin, glm::vec3 dir,
                                tinyobj::attrib_t const& attrib,
                                tinyobj::shape_t const& shape
                                );

// Determines whether the segment `origin -> origin + dir` intersects any triangles or not.
bool intersects_any(
                    glm::vec3 origin, glm::vec3 dir,
                    tinyobj::attrib_t const& attrib,
                    tinyobj::shape_t const& shape
                    );

std::vector<glm::vec4> sample_points(
                                     float x_min, float x_max, float x_step,
                                     float y_min, float y_max, float y_step,
                                     float z_min, float z_max, float z_step,
                                     std::vector<tinyobj::shape_t> const& shapes,
                                     tinyobj::attrib_t const& attrib
                                     );

std::vector<std::pair<int, int>> form_connections(
                                                  int offset, int count,
                                                  objscan_position const* positions, int N,
                                                  float step_x, float step_y, float step_z,
                                                  std::vector<tinyobj::shape_t> const& shapes,
                                                  tinyobj::attrib_t const& attrib
                                                  );

void calculate_bounding_box(
                            float& x_min, float& y_min, float& z_min,
                            float& x_max, float& y_max, float& z_max,
                            tinyobj::attrib_t const& attrib);