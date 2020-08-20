// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: Centripetal Catmull-Rom spline implementation
//

#pragma once
#include <cassert>
#include <type_traits>
#include <vector>

#define GENSTEP(asgn, t, t0, t1, p0, p1) \
Point const asgn([=]() { \
    float const tmp0 = (t1 - t) / (t1 - t0); \
    float const tmp1 = (t - t0) / (t1 - t0); \
    float const x = tmp0 * GetX(p0) + tmp1 * GetX(p1); \
    float const y = tmp0 * GetY(p0) + tmp1 * GetY(p1); \
    float const z = tmp0 * GetZ(p0) + tmp1 * GetZ(p1); \
    return Point(x, y, z); \
}());

struct Interpolated_User_Data {
    uint64_t user0; float weight0;
    uint64_t user1; float weight1;
};

class Float_Iterator {
public:
    constexpr Float_Iterator(float v) : it(v) {}
    bool operator!=(Float_Iterator const& other) const {
        return it < other.it;
    }

    Float_Iterator& operator++() {
        it += 0.001f;
        return *this;
    }

    float operator*() const {
        return it;
    }
private:
    float it;
};

template<typename Point>
class Catmull_Rom {
public:
    Catmull_Rom(Point const& p0, Point const& p1, Point const& p2, Point const& p3, uint64_t user1 = 0, uint64_t user2 = 0)
        : p{ p0, p1, p2, p3 }, user1(user1), user2(user2) {
        t[0] = GetT(0.0f, p0, p1);
        t[1] = GetT(t[0], p1, p2);
        t[2] = GetT(t[1], p2, p3);
    }

    void GeneratePoints(size_t unCountPoints, Point* pOutBuf) const {
        size_t i = 0;
        float const flStep = (t[1] - t[0]) / unCountPoints;
        for (float T = t[0]; T < t[1] && i < unCountPoints; T += flStep) {
            // If we're filling the last slot in the out array, then emit
            // the endpoint of the curve rather than the next one.
            if (i == unCountPoints - 1) {
                T = t[1];
            }

            GENSTEP(a1, T, 0.0f, t[0], p[0], p[1]);
            GENSTEP(a2, T, t[0], t[1], p[1], p[2]);
            GENSTEP(a3, T, t[1], t[2], p[2], p[3]);
            GENSTEP(b1, T, 0.0f, t[1], a1, a2);
            GENSTEP(b2, T, t[0], t[2], a2, a3);
            GENSTEP(c0, T, t[0], t[1], b1, b2);
            pOutBuf[i++] = c0;
        }
    }

    std::vector<Point> GeneratePoints(size_t unSubdivisions, std::vector<Interpolated_User_Data>& aUserData) const {
        std::vector<Point> ret;

        float const flStep = (t[1] - t[0]) / (unSubdivisions + 1);
        for (float T = t[0]; LessThanOrWithinEpsilon(T, t[1], 0.00001f); T += flStep) {
            GENSTEP(a1, T, 0.0f, t[0], p[0], p[1]);
            GENSTEP(a2, T, t[0], t[1], p[1], p[2]);
            GENSTEP(a3, T, t[1], t[2], p[2], p[3]);
            GENSTEP(b1, T, 0.0f, t[1], a1, a2);
            GENSTEP(b2, T, t[0], t[2], a2, a3);
            GENSTEP(c0, T, t[0], t[1], b1, b2);

            auto w0 = (T - t[0]) / (t[1] - t[0]);
            auto w1 = 1 - w0;
            ret.push_back(c0);
            aUserData.push_back({ user1, w0, user2, w1 });
        }

        assert(ret.size() == unSubdivisions + 2);

        return ret;
    }

    Point operator()(float const T) const {
        GENSTEP(a1, T, 0.0f, t[0], p[0], p[1]);
        GENSTEP(a2, T, t[0], t[1], p[1], p[2]);
        GENSTEP(a3, T, t[1], t[2], p[2], p[3]);
        GENSTEP(b1, T, 0.0f, t[1], a1, a2);
        GENSTEP(b2, T, t[0], t[2], a2, a3);
        GENSTEP(c0, T, t[0], t[1], b1, b2);
        return c0;
    }

    auto begin() const { return Float_Iterator(t[0]); }
    auto end() const { return Float_Iterator(t[1]); }
private:
    bool LessThanOrWithinEpsilon(float flLhs, float flRhs, float flEpsilon) const {
        float flDist = fabs(flRhs - flLhs);
        return flLhs < flRhs || (flDist < flEpsilon);
    }

    float GetT(float t, Point const& p0, Point const& p1) {
        float const dx = GetX(p1) - GetX(p0);
        float const dy = GetY(p1) - GetY(p0);
        float const dz = GetZ(p1) - GetZ(p0);
        float const a = dx * dx + dy * dy + dz * dz;
        float const b = sqrt(a);
        float const c = sqrt(b);
        return c + t;
    }
private:
    float t[3];
    Point p[4];
    uint64_t user1, user2;
};

template<typename Point>
class Catmull_Rom_Composite {
public:
    Catmull_Rom_Composite() : m_unPoints(0), m_pPoints(0) {}
    Catmull_Rom_Composite(size_t unPoints, Point const* pPoints, uint64_t const* pUser)
        : m_unPoints(unPoints), m_pPoints(pPoints), m_pUser(pUser) {}

    void GeneratePoints(size_t unBufCount, Point* pOutBuf) const {
        assert(m_unPoints >= 4);
        assert(m_pPoints != NULL);

        auto const unCurveCount = m_unPoints - 3;
        auto const unPointsPerCurve = unBufCount / unCurveCount;
        auto const unRemainder = unBufCount - unPointsPerCurve * unCurveCount;

        for (size_t iCurve = 0; iCurve < unCurveCount; iCurve++) {
            bool const bLast = (iCurve == unCurveCount - 1);
            // Add remainder points to the point count when processing the last curve
            auto const unPoints = bLast ? unPointsPerCurve + unRemainder : unPointsPerCurve;
            auto const cr = Catmull_Rom<Point>(m_pPoints[iCurve + 0], m_pPoints[iCurve + 1], m_pPoints[iCurve + 2], m_pPoints[iCurve + 3]);
            auto const uiBaseOffset = iCurve * unPointsPerCurve;
            cr.GeneratePoints(unPoints, &pOutBuf[uiBaseOffset]);
        }
    }

    std::vector<Point> GeneratePoints(size_t unSubdivisions, std::vector<Interpolated_User_Data>& aUserData) const {
        assert(m_unPoints >= 4);
        assert(m_pPoints != NULL);
        std::vector<Point> ret;

        auto const unCurveCount = m_unPoints - 3;
        for (size_t iCurve = 0; iCurve < unCurveCount; iCurve++) {
            auto const cr = Catmull_Rom<Point>(m_pPoints[iCurve + 0], m_pPoints[iCurve + 1], m_pPoints[iCurve + 2], m_pPoints[iCurve + 3], m_pUser[iCurve + 1], m_pUser[iCurve + 2]);
            std::vector<Interpolated_User_Data> aUserDataBuffer;
            auto const res = cr.GeneratePoints(unSubdivisions, aUserDataBuffer);
            ret.insert(ret.end(), res.cbegin(), res.cend() - 1);
            aUserData.insert(aUserData.end(), aUserDataBuffer.cbegin(), aUserDataBuffer.cend() - 1);
        }

        assert(ret.size() == aUserData.size());

        return ret;
    }

private:
    size_t m_unPoints;
    Point const* m_pPoints;
    uint64_t const* m_pUser;
};
