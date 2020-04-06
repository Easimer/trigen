// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: wood trunk mesh generation
//

#define _USE_MATH_DEFINES
#include <cmath>
#include <trigen/linear_math.h>
#include <trigen/catmull_rom.h>
#include "trunk_generator.h"

Mesh_Builder::Optimized_Mesh MeshFromSpline(Catmull_Rom<lm::Vector4> const& cr) {
    size_t const unPoints = 16; // Number of points in the spline
    size_t const unSides = 12; // Number of cylinder sides
    Mesh_Builder mb(0.00001f);
    auto const avPoints = [unPoints, cr]() {
        std::vector<lm::Vector4> buf;
        buf.resize(unPoints);
        cr.GeneratePoints(unPoints, buf.data());
        return buf;
    }();

    auto const flHalfAlpha = M_PI / unSides;
    auto const flAlpha = 2 * flHalfAlpha;
    auto const matRotateY = lm::RotationY(flAlpha);
    auto const flBaseRadius = 4.0f;

    for (size_t i = 0; i < unPoints - 1; i++) {
        auto const& sp0 = avPoints[i + 0];
        auto const& sp1 = avPoints[i + 1];
        auto fwd = lm::Vector4(1, 0, 0);
        auto rgt = lm::Vector4(0, 0, 1);
        auto const flRadius0 = flBaseRadius * powf(0.9, i + 0);
        auto const flRadius1 = flBaseRadius * powf(0.9, i + 1);
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
            rgt = matRotateY * rgt;
        }
        fwd = matRotateY * fwd;
    }

    auto opt = mb.Optimize();
    return opt;
}
