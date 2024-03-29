// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: Linear math types
//
// Based on the LENS Engine math library
// https://git.easimer.net/lens/tree/public/math
//

#pragma once
#include <cmath>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <immintrin.h>

namespace lm {
    class Matrix4 {
    public:
        // Creates a null matrix
        constexpr Matrix4() noexcept : m_flValues{} {}

        // Creates a scalar matrix
        constexpr Matrix4(float diag) noexcept
            : m_flValues {
            diag, 0, 0, 0,
            0, diag, 0, 0,
            0, 0, diag, 0,
            0, 0, 0, diag,
        } {}

        // Creates a matrix filled with the values in the supplied
        // array, in column-major order.
        constexpr Matrix4(float const m[16])
            : m_flValues{
            m[0], m[1], m[2], m[3],
            m[4], m[5], m[6], m[7],
            m[8], m[9], m[10], m[11],
            m[12], m[13], m[14], m[15],
        } {}

        float const* Data() const {
            return m_flValues;
        }

        float* Data() {
            return m_flValues;
        }

        float Idx(unsigned uiRow, unsigned uiCol) const {
            uiRow &= 0b11;
            uiCol &= 0b11;
            return m_flValues[uiCol * 4 + uiRow];
        }

        float& Idx(unsigned uiRow, unsigned uiCol) {
            uiRow &= 0b11;
            uiCol &= 0b11;
            return m_flValues[uiCol * 4 + uiRow];
        }

        float m_flValues[16];
    };

    class Vector4 {
    public:
        constexpr Vector4() : m_flValues{ 0, 0, 0, 0 } {}
        constexpr Vector4(float x, float y, float z, float w = 0.0f)
            : m_flValues{ x, y, z, w } {}

        float operator[](unsigned idx) const {
            return m_flValues[idx & 3];
        }

        Vector4 operator+(Vector4 const& other) const {
            return {
                m_flValues[0] + other[0],
                m_flValues[1] + other[1],
                m_flValues[2] + other[2],
                m_flValues[3] + other[3],
            };
        }

        float m_flValues[4];
    };

    inline Vector4 operator*(float lam, Vector4 const& other) {
        return {
            lam * other[0],
            lam * other[1],
            lam * other[2],
            lam * other[3],
        };
    }

    inline Vector4 operator/(Vector4 const& lhs, float rhs) {
        return { 
            lhs[0] / rhs,
            lhs[1] / rhs,
            lhs[2] / rhs,
            lhs[3] / rhs,
        };
    }

    inline Vector4 operator-(Vector4 const& lhs, Vector4 const& rhs) {
        return lhs + (-1 * rhs);
    }

    inline Vector4 operator-(Vector4 const& v) {
        return Vector4(-v[0], -v[1], -v[2], -v[3]);
    }

    using mat4 = Matrix4;

    namespace detail {
        // Matrix multiplication common stuff
        inline void matmul_implcore(Matrix4 const& A, __m128 col[4], __m128 sum[4]) {
            sum[0] = _mm_setzero_ps();
            sum[0] = _mm_fmadd_ps(_mm_set1_ps(A.m_flValues[0 * 4 + 0]), col[0], sum[0]);
            sum[0] = _mm_fmadd_ps(_mm_set1_ps(A.m_flValues[0 * 4 + 1]), col[1], sum[0]);
            sum[0] = _mm_fmadd_ps(_mm_set1_ps(A.m_flValues[0 * 4 + 2]), col[2], sum[0]);
            sum[0] = _mm_fmadd_ps(_mm_set1_ps(A.m_flValues[0 * 4 + 3]), col[3], sum[0]);

            sum[1] = _mm_setzero_ps();
            sum[1] = _mm_fmadd_ps(_mm_set1_ps(A.m_flValues[1 * 4 + 0]), col[0], sum[1]);
            sum[1] = _mm_fmadd_ps(_mm_set1_ps(A.m_flValues[1 * 4 + 1]), col[1], sum[1]);
            sum[1] = _mm_fmadd_ps(_mm_set1_ps(A.m_flValues[1 * 4 + 2]), col[2], sum[1]);
            sum[1] = _mm_fmadd_ps(_mm_set1_ps(A.m_flValues[1 * 4 + 3]), col[3], sum[1]);

            sum[2] = _mm_setzero_ps();
            sum[2] = _mm_fmadd_ps(_mm_set1_ps(A.m_flValues[2 * 4 + 0]), col[0], sum[2]);
            sum[2] = _mm_fmadd_ps(_mm_set1_ps(A.m_flValues[2 * 4 + 1]), col[1], sum[2]);
            sum[2] = _mm_fmadd_ps(_mm_set1_ps(A.m_flValues[2 * 4 + 2]), col[2], sum[2]);
            sum[2] = _mm_fmadd_ps(_mm_set1_ps(A.m_flValues[2 * 4 + 3]), col[3], sum[2]);

            sum[3] = _mm_setzero_ps();
            sum[3] = _mm_fmadd_ps(_mm_set1_ps(A.m_flValues[3 * 4 + 0]), col[0], sum[3]);
            sum[3] = _mm_fmadd_ps(_mm_set1_ps(A.m_flValues[3 * 4 + 1]), col[1], sum[3]);
            sum[3] = _mm_fmadd_ps(_mm_set1_ps(A.m_flValues[3 * 4 + 2]), col[2], sum[3]);
            sum[3] = _mm_fmadd_ps(_mm_set1_ps(A.m_flValues[3 * 4 + 3]), col[3], sum[3]);
        }

        // Matrix multiplication where B is aligned
        inline Matrix4 matmul_aligned(Matrix4 const& A, Matrix4 const& B) {
            Matrix4 m;
            __m128 col[4], sum[4];

            col[0] = _mm_load_ps(B.m_flValues + 0);
            col[1] = _mm_load_ps(B.m_flValues + 4);
            col[2] = _mm_load_ps(B.m_flValues + 8);
            col[3] = _mm_load_ps(B.m_flValues + 12);

            matmul_implcore(A, col, sum);

            _mm_storeu_ps(m.m_flValues + 0, sum[0]);
            _mm_storeu_ps(m.m_flValues + 4, sum[1]);
            _mm_storeu_ps(m.m_flValues + 8, sum[2]);
            _mm_storeu_ps(m.m_flValues + 12, sum[3]);

            return m;
        }

        // Matrix multiplication where B is unaligned 
        inline Matrix4 matmul_unaligned(Matrix4 const& A, Matrix4 const& B) {
            Matrix4 m;
            __m128 col[4], sum[4];

            col[0] = _mm_loadu_ps(B.m_flValues + 0);
            col[1] = _mm_loadu_ps(B.m_flValues + 4);
            col[2] = _mm_loadu_ps(B.m_flValues + 8);
            col[3] = _mm_loadu_ps(B.m_flValues + 12);

            matmul_implcore(A, col, sum);

            _mm_storeu_ps(m.m_flValues + 0, sum[0]);
            _mm_storeu_ps(m.m_flValues + 4, sum[1]);
            _mm_storeu_ps(m.m_flValues + 8, sum[2]);
            _mm_storeu_ps(m.m_flValues + 12, sum[3]);

            return m;
        }
    }

    // Creates a translation matrix
    inline Matrix4 Translation(float x, float y, float z) {
        float const m[16] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1 };
        return Matrix4(m);
    }

    inline Matrix4 Translation(Vector4 const& v) {
        return Translation(v[0], v[1], v[2]);
    }

    inline Matrix4 operator*(Matrix4 const& A, Matrix4 const& B) {
        if (((size_t)B.Data() & 15) == 0) {
            return detail::matmul_aligned(A, B);
        } else {
            return detail::matmul_unaligned(A, B);
        }
    }

    inline void Perspective(Matrix4& forward, Matrix4& inverse, float width, float height, float fov, float flNear, float flFar) {
        float aspect = height / width;
        float e = 1.0f / tan(fov / 2.0f);

        // TODO(danielm): when doing unity builds, something #defines
        // `far` and `near` to something that causes these lines to become
        // syntactically invalid. Renaming them to flNear and flFar
        // fixed this, but we should find out who does the defs.
        float F = ((flFar + flNear) / (flNear - flFar));
        float f = (2 * flFar * flNear) / (flNear - flFar);

        float m[16] = {
            e, 0, 0, 0,
            0, e / aspect, 0, 0,
            0, 0, F, -1.0f,
            0, 0, f, 0,
        };

        float minv[16] = {
            1 / e, 0, 0, 0,
            0, aspect / e, 0, 0,
            0, 0, 0, 1 / f,
            0, 0, -1, F / f
        };
        forward = Matrix4(m);
        inverse = Matrix4(minv);
    }

    inline Matrix4 RotationY(float radians) {
        Matrix4 ret(1.0f);
        auto const s = sinf(radians);
        auto const c = cosf(radians);

        ret.Idx(0, 0) = c;
        ret.Idx(2, 0) = -s;
        ret.Idx(0, 2) = s;
        ret.Idx(2, 2) = c;

        return ret;
    }

    inline Matrix4 Scale(float x, float y, float z) {
        Matrix4 ret(1.0f);

        ret.Idx(0, 0) = x;
        ret.Idx(1, 1) = y;
        ret.Idx(2, 2) = z;

        return ret;
    }

    inline Matrix4 Scale(float l) {
        return Scale(l, l, l);
    }

    inline Vector4 operator*(Matrix4 const& lhs, Vector4 const& rhs) {
        float const* pMat = lhs.Data();
        float const* pVec = rhs.m_flValues;
        float const v0 = pVec[0];
        float const v1 = pVec[1];
        float const v2 = pVec[2];
        float const v3 = pVec[3];
        float r[4];
        for (int i = 0; i < 4; i++) {
            float const c0 = pMat[i + 0 * 4];
            float const c1 = pMat[i + 1 * 4];
            float const c2 = pMat[i + 2 * 4];
            float const c3 = pMat[i + 3 * 4];

            r[i] = c0 * v0 + c1 * v1 + c2 * v2 + c3 * v3;
        }
        return Vector4(r[0], r[1], r[2], r[3]);
    }

    inline float LengthSq(Vector4 const& v) {
        return v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
    }

    inline float Length(Vector4 const& v) {
        return sqrtf(LengthSq(v));
    }

    inline Vector4 Normalized(Vector4 const& v) {
        auto const L = Length(v);
        return Vector4(v[0] / L, v[1] / L, v[2] / L, v[3] / L);
    }

    inline Vector4 Cross(Vector4 const& u, Vector4 const& v) {
        return Vector4(u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]);
    }
}

inline float GetX(lm::Vector4 const& v) { return v[0]; }
inline float GetY(lm::Vector4 const& v) { return v[1]; }
inline float GetZ(lm::Vector4 const& v) { return v[2]; }
