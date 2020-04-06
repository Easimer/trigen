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
    size_t const unPoints = 8; // Number of points in the spline
    size_t const unSides = 5; // Number of cylinder sides
    Mesh_Builder mb(0.0001f);
    lm::Vector4 avPoints[unPoints];
    cr.GeneratePoints(unPoints, avPoints);

    auto const flHalfAlpha = M_PI / unSides;
    auto const flAlpha = 2 * flHalfAlpha;
    auto const matRotateY = lm::RotationY(flAlpha);
    auto const flRadius = 1.0f;
    auto const flSideLength = 2 * flRadius * cosf(flHalfAlpha);


    for (size_t i = 0; i < unPoints - 1; i++) {
        auto const& sp0 = avPoints[i + 0];
        auto const& sp1 = avPoints[i + 1];
        auto const delta = sp1 - sp0;
        auto fwd = lm::Vector4(1, 0, 0);
        auto rgt = lm::Vector4(0, 0, 1);

        auto cur = sp0 + flRadius * fwd;
        for (size_t iSide = 0; iSide < unSides; iSide++) {
            auto const p1 = cur + delta;
            auto const p2 = cur + flSideLength * rgt;
            // auto const p3 = p2;
            // auto const p4 = p1;
            // auto const p5 = p3 + delta;
            auto const p5 = p2 + delta;

            mb.PushTriangle(cur, p1, p2);
            mb.PushTriangle(p2, p1, p5);


            cur = p2;
            rgt = matRotateY * rgt;
        }
        fwd = matRotateY * fwd;
    }

    return mb.Optimize();
}
