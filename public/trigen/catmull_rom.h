// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: Centripetal Catmull-Rom spline implementation
//

#include <type_traits>

#define GENSTEP(asgn, t, t0, t1, p0, p1) \
Point const asgn = [=]() { \
    float const tmp0 = (t1 - t) / (t1 - t0); \
    float const tmp1 = (t - t0) / (t1 - t0); \
    float const x = tmp0 * GetX(p0) + tmp1 * GetX(p1); \
    float const y = tmp0 * GetY(p0) + tmp1 * GetY(p1); \
    return Point(x, y); \
}();

template<typename Point>
class Catmull_Rom {
public:
    Catmull_Rom(Point const& p0, Point const& p1, Point const& p2, Point const& p3)
        : p{ p0, p1, p2, p3 } {
        t[0] = GetT(0.0f, p0, p1);
        t[1] = GetT(t[0], p1, p2);
        t[2] = GetT(t[1], p2, p3);
    }

    void GeneratePoints(size_t unCountPoints, Point* pOutBuf) {
        size_t i = 0;
        float const flStep = (t[1] - t[0]) / unCountPoints;
        for (float T = t[0]; T < t[1]; T += flStep) {
            GENSTEP(a1, T, 0.0f, t[0], p[0], p[1]);
            GENSTEP(a2, T, t[0], t[1], p[1], p[2]);
            GENSTEP(a3, T, t[1], t[2], p[2], p[3]);
            GENSTEP(b1, T, 0.0f, t[1], a1, a2);
            GENSTEP(b2, T, t[0], t[2], a2, a3);
            GENSTEP(c0, T, t[0], t[1], b1, b2);
            pOutBuf[i++] = c0;
        }
    }
private:

    float GetT(float t, Point const& p0, Point const& p1) {
        float const dx = GetX(p1) - GetX(p0);
        float const dy = GetY(p1) - GetY(p0);
        float const a = dx * dx + dy * dy;
        float const b = sqrt(a);
        float const c = sqrt(b);
        return c + t;
    }
private:
    float t[3];
    Point p[4];
};
