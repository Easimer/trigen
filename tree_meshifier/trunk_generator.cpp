// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: wood trunk mesh generation
//

#define _USE_MATH_DEFINES
#include <cmath>
#include <trigen/linear_math.h>
#include <trigen/catmull_rom.h>
#include <trigen/tree_meshifier.h>
#include "trunk_generator.h"

struct Mesh_From_Spline_Params {
    std::vector<lm::Vector4> const& avPoints;
    std::vector<Interpolated_User_Data> const& aUserData;
    size_t const unPoints;
    size_t const unSides;
};

static lm::Vector4 Perpendicular(lm::Vector4 const& v) {
    auto v0 = lm::Vector4(v[2], v[2], -v[0] - v[1]);
    auto v1 = lm::Vector4(-v[1]-v[2], v[0], v[0]);
    auto idx = (v[2] != 0) && (-v[0] != v[1]);
    return idx ? v1 : v0;
}

static Optimized_Mesh<TG_Vertex> MeshFromSpline_impl(Mesh_From_Spline_Params const& params, TG_RadiusFunc radiusFunc) {
    auto const& avPoints = params.avPoints;
    auto const unPoints = params.unPoints;
    auto const unSides = params.unSides;
    auto const& aUserData = params.aUserData;

    Mesh_Builder<TG_Vertex> mb(0.00001f);

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
        auto& interp0 = aUserData[i + 0];
        auto& interp1 = aUserData[i + 1];
        auto const flRadius0 = radiusFunc(i + 0, sp0, interp0.user0, interp0.weight0, interp0.user1, interp0.weight1);
        auto const flRadius1 = radiusFunc(i + 1, sp1, interp1.user0, interp1.weight0, interp1.user1, interp1.weight1);
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

            float t0 = Length(p0 - sp0) / Length(up);
            float t1 = Length(p1 - sp0) / Length(up);
            float t2 = Length(p2 - sp0) / Length(up);
            float t5 = Length(p5 - sp0) / Length(up);

            float x0 = (iSide + 0) / (float)unSides;
            float x1 = (iSide + 1) / (float)unSides;

            TG_Vertex v0 = { { p0[0], p0[1], p0[2], p0[3] }, { x0, t0 } };
            TG_Vertex v1 = { { p1[0], p1[1], p1[2], p1[3] }, { x0, t1 } };
            TG_Vertex v2 = { { p2[0], p2[1], p2[2], p2[3] }, { x1, t2 } };
            TG_Vertex v5 = { { p5[0], p5[1], p5[2], p5[3] }, { x1, t5 } };

            mb.PushTriangle(v0, v1, v2);
            mb.PushTriangle(v2, v1, v5);

            starter0 = p2;
            starter1 = p5;
            fwd = matRotateY * fwd;
            rgt = lm::Cross(fwd, up);
        }
        // fwd = matRotateY * fwd;
    }

    return mb.Optimize();
}

static float LegacyRadiusFunction(size_t i, lm::Vector4 const& p, uint64_t user0, float weight0, uint64_t user1, float weight1) {
    return 4.0f * powf(0.99, i + 0);
}

Optimized_Mesh<TG_Vertex> MeshFromSpline(Catmull_Rom_Composite<lm::Vector4> const& cr, TG_RadiusFunc const& radiusFunc) {
    size_t const unSubdivisions = 8; // Number of subdivisions in the spline
    size_t const unSides = 5; // Number of cylinder sides
    Mesh_Builder<TG_Vertex> mb(0.00001f);
    /*
    auto const avPoints = [unPoints, cr]() {
        std::vector<lm::Vector4> buf;
        buf.resize(unPoints);
        cr.GeneratePoints(unPoints, buf.data());
        return buf;
    }();
    */
    std::vector<Interpolated_User_Data> aUserData;
    auto const avPoints = cr.GeneratePoints(unSubdivisions, aUserData);

    return MeshFromSpline_impl({ avPoints, aUserData, avPoints.size(), unSides }, radiusFunc);
}
