// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: wood trunk mesh generation declarations
//

#pragma once

#include <functional>
#include <trigen/catmull_rom.h>
#include <trigen/linear_math.h>
#include <trigen/meshbuilder.h>

// =============================================
// Purpose: a callback that calculates the radius
// around a given point in the curve.
// =============================================
using TG_RadiusFunc = std::function<float(size_t iPoint, lm::Vector4 const& vPoint)>;

// =============================================
// Purpose: build a 3D mesh from a Catmull-Rom spline.
// =============================================
Mesh_Builder::Optimized_Mesh MeshFromSpline(Catmull_Rom<lm::Vector4> const& cr);
Mesh_Builder::Optimized_Mesh MeshFromSpline(Catmull_Rom_Composite<lm::Vector4> const& cr);
Mesh_Builder::Optimized_Mesh MeshFromSpline(Catmull_Rom_Composite<lm::Vector4> const& cr, TG_RadiusFunc radiusFunc);

