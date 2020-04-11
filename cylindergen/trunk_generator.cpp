// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: wood trunk mesh generation
//

#include "stdafx.h"
#define _USE_MATH_DEFINES
#include <cmath>
#include <trigen/linear_math.h>
#include <trigen/catmull_rom.h>
#include "trunk_generator.h"

struct Mesh_From_Spline_Params {
    std::vector<lm::Vector4> const& avPoints;
    size_t const unPoints;
    size_t const unSides;
};

static lm::Vector4 Perpendicular(lm::Vector4 const& v) {
    auto v0 = lm::Vector4(v[2], v[2], -v[0] - v[1]);
    auto v1 = lm::Vector4(-v[1]-v[2], v[0], v[0]);
    auto idx = (v[2] != 0) && (-v[0] != v[1]);
    return idx ? v1 : v0;
}

static Mesh_Builder::Optimized_Mesh MeshFromSpline_impl(Mesh_From_Spline_Params const& params, TG_RadiusFunc radiusFunc) {
    auto const& avPoints = params.avPoints;
    auto const unPoints = params.unPoints;
    auto const unSides = params.unSides;

    Mesh_Builder mb(0.00001f);

    auto const flHalfAlpha = M_PI / unSides;
    auto const flAlpha = 2 * flHalfAlpha;
    auto const matRotateY = lm::RotationY(flAlpha);
    auto const flBaseRadius = 4.0f;

    for (size_t i = 0; i < unPoints - 1; i++) {
        auto const& sp0 = avPoints[i + 0];
        auto const& sp1 = avPoints[i + 1];
        auto const up = lm::Normalized(sp1 - sp0);
        auto fwd = lm::Vector4(1, 0, 0);
        auto rgt = lm::Cross(fwd, up);
        // auto fwd = lm::Vector4(1, 0, 0);
        // auto rgt = lm::Vector4(0, 0, 1);
        auto const flRadius0 = radiusFunc(i, sp0);
        auto const flRadius1 = radiusFunc(i, sp0);
        auto const flSideLength0 = 2 * flRadius0 * cosf(flHalfAlpha);
        auto const flSideLength1 = 2 * flRadius1 * cosf(flHalfAlpha);

        lm::Vector4 starter0 = sp0 + flRadius0 * fwd;
        lm::Vector4 starter1 = sp1 + flRadius1 * fwd;
        for (size_t iSide = 0; iSide < unSides; iSide++) {
            auto const& p0 = starter0;
            auto const& p1 = starter1;
            auto const p2 = p0 + flSideLength0 * rgt;
            // auto const p3 = p2;
            // auto const p4 = p1;
            auto const p5 = p1 + flSideLength1 * rgt;

            mb.PushTriangle(p0, p1, p2);
            mb.PushTriangle(p2, p1, p5);

            starter0 = p2;
            starter1 = p5;
            fwd = matRotateY * fwd;
            rgt = lm::Cross(fwd, up);
        }
        // fwd = matRotateY * fwd;
    }

    return mb.Optimize();
}

static float LegacyRadiusFunction(size_t i, lm::Vector4 const& p) {
    return 4.0f * powf(0.99, i + 0);
}

Mesh_Builder::Optimized_Mesh MeshFromSpline(Catmull_Rom<lm::Vector4> const& cr) {
    size_t const unPoints = 8; // Number of points in the spline
    size_t const unSides = 6; // Number of cylinder sides
    auto const avPoints = [unPoints, cr]() {
        std::vector<lm::Vector4> buf;
        buf.resize(unPoints);
        cr.GeneratePoints(unPoints, buf.data());
        return buf;
    }();

    return MeshFromSpline_impl({ avPoints, unPoints, unSides }, LegacyRadiusFunction);
}

Mesh_Builder::Optimized_Mesh MeshFromSpline(Catmull_Rom_Composite<lm::Vector4> const& cr) {
    return MeshFromSpline(cr, LegacyRadiusFunction);
}

Mesh_Builder::Optimized_Mesh MeshFromSpline(Catmull_Rom_Composite<lm::Vector4> const& cr, TG_RadiusFunc radiusFunc) {
    size_t const unSubdivisions = 8; // Number of subdivisions in the spline
    size_t const unSides = 5; // Number of cylinder sides
    Mesh_Builder mb(0.00001f);
    /*
    auto const avPoints = [unPoints, cr]() {
        std::vector<lm::Vector4> buf;
        buf.resize(unPoints);
        cr.GeneratePoints(unPoints, buf.data());
        return buf;
    }();
    */
    auto const avPoints = cr.GeneratePoints(unSubdivisions);

    return MeshFromSpline_impl({ avPoints, avPoints.size(), unSides }, radiusFunc);
}
