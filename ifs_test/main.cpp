// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: Iterated function system test
//

#include <cstdio>
#include <cassert>
#include <ctime>
#include <type_traits>
#include <array>
#include <trigen/sdl_helper.h>
#include <SDL_ttf.h>

static float randf() {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

template<typename T>
struct Vector2 {
    constexpr Vector2() : x(T(0)), y(T(0)) {}
    constexpr Vector2(T x, T y) : x(x), y(y) {}
    T x, y;
};

template<typename T> T GetX(Vector2<T> const& v) { return v.x; }
template<typename T> T GetY(Vector2<T> const& v) { return v.y; }
template<typename T> T GetZ(Vector2<T> const& v) { return T(0); }

using Vector2f = Vector2<float>;

struct ScaleRotateTranslate2 {
    float a, b;
    float c, d;
    float tx, ty;

    using domain_type = Vector2f;

    constexpr ScaleRotateTranslate2()
    : a(1), b(0),
    c(0), d(1),
    tx(0), ty(0) {
    }

    constexpr ScaleRotateTranslate2(float sx, float sy)
    : a(sx), b(0), c(0), d(sy), tx(0), ty(0) {}

    constexpr ScaleRotateTranslate2(float sx, float sy, float tx, float ty)
    : a(sx), b(0), c(0), d(sy), tx(tx), ty(ty) {}

    ScaleRotateTranslate2(float deg, float sx, float sy, float tx, float ty)
    : tx(tx), ty(ty) {
        a = sx * cosf(deg);
        b = -sy * sinf(deg);
        c = sx * sinf(deg);
        d = sy * cosf(deg);
    }

    constexpr ScaleRotateTranslate2(float a, float b, float c, float d, float tx, float ty)
    : a(a), b(b),
      c(c), d(d), 
      tx(tx), ty(ty) {
    }

    Vector2f operator()(Vector2f const& x) const {
        return Vector2f(a * x.x + b * x.y + tx, c * x.x + d * x.y + ty);
    }
};

template<typename T>
class IIterated_Function_System {
public:
    virtual T apply(T const& x) const = 0;
};

template<typename T, size_t N>
struct Iterated_Function_System : public IIterated_Function_System<typename T::domain_type> {
    static_assert(N > 0, "There must be atleast one transform!");

    std::array<T, N> const transforms;
    std::array<float, N> const probabilities;
    using domain_type = typename T::domain_type;

    constexpr Iterated_Function_System(std::array<T, N>&& t, std::array<float, N>&& p)
        : transforms(std::move(t)), probabilities(std::move(p)) {}

    domain_type apply(domain_type const& x) const override {
        float const p = randf();

        size_t i = 0;
        float c = 0;
        while (i < N) {
            c += probabilities[i];
            if (p < c) {
                return transforms[i](x);
            }
            i++;
        }

        return transforms[N - 1](x);
    }
};

int main(int argc, char** argv) {
    Iterated_Function_System<ScaleRotateTranslate2, 4> Sierpinski = {
        {
            ScaleRotateTranslate2(0.5, 0.0, 0.0, 0.5, 0, 0),
            ScaleRotateTranslate2(0.5, 0.0, 0.0, 0.5, 128, 0),
            ScaleRotateTranslate2(0.5, 0.0, 0.0, 0.5, 128, 128),
            ScaleRotateTranslate2(0.5, 0.0, 0.0, 0.5, 128, 128),
        },
        {0.15, 0.35, 0.35, 0.15},
    };

    Iterated_Function_System<ScaleRotateTranslate2, 2> Waves = {
        {
            ScaleRotateTranslate2(0.75, 0.0, 0.0, 0.75, 64, 64),
            ScaleRotateTranslate2(0.75, 0.2, -0.2, 0.75, -32, 32),
        },
        {0.618033989f, 0.381966011f},
    };

    Iterated_Function_System<ScaleRotateTranslate2, 4> Tree = {
        {
            ScaleRotateTranslate2(0.29, 0.4, -0.4, 0.3, 0.28, 0.44),
            ScaleRotateTranslate2(0.33, -0.34, 0.39, 0.4, 0.41, 0),
            ScaleRotateTranslate2(0.42, 0.0, 0.0, 0.63, 0.29, 0.36),
            ScaleRotateTranslate2(0.61, 0.0, 0.0, 0.61, 0.19, 0.23),
        },
        {0.25, 0.25, 0.25, 0.25},
    };

    Iterated_Function_System<ScaleRotateTranslate2, 4> Spiral = {
        {
            ScaleRotateTranslate2(0.5, 0.0, 0.0, 0.5, 64, 64),
            ScaleRotateTranslate2(0.5, 0.25, 0.25, 0.5, -64, 64),
            ScaleRotateTranslate2(0.5, -0.25, 0.25, 0.5, -64, 64),
            ScaleRotateTranslate2(0.5, 0.25, -0.25, 0.5, -64, 64),
        },
        {0.618033989f, 0.190983005, 0.0954915025, 0.0954915025},
    };

    std::array<std::string, 4> titles = { "Spiral", "Sierpisnki Triangle", "Wave", "Tree" };
    std::array<IIterated_Function_System<Vector2f>*, 4> systems = { &Spiral, &Sierpinski, &Waves, &Tree };
    std::array<float, 4> zoom = { 2.0, 1.0, 1.0, 512.0 };


    srand(time(NULL));
    SDL_Init(SDL_INIT_VIDEO);
    TTF_Init();
    sdl::Renderer r("Stochastic Iterated Function System", 1024, 1024);
    if (r) {
        auto hFont = TTF_OpenFont("iosevka-regular.ttf", 24);
        for (size_t i = 0; i < systems.size(); i = (i + 1) % systems.size()) {
            auto& S = systems[i];
            auto& T = titles[i];

            // Clear screen
            SDL_SetRenderDrawColor(r, 0, 0, 0, 255);
            SDL_RenderClear(r);
            
            // Draw title
            if (hFont != NULL) {
                SDL_Surface* pTextSurface = TTF_RenderText_Solid(hFont, titles[i].c_str(), { 255, 255, 255, 255 });
                if (pTextSurface != NULL) {
                    auto pTex = SDL_CreateTextureFromSurface(r, pTextSurface);
                    if (pTex != NULL) {
                        SDL_Rect rect{ 0, 0, pTextSurface->w, pTextSurface->h };
                        SDL_RenderCopy(r, pTex, &rect, &rect);
                        SDL_DestroyTexture(pTex);
                    }
                    SDL_FreeSurface(pTextSurface);
                }
            }

            SDL_SetRenderDrawColor(r, 0, 255, 0, 255);
            Vector2f v(0, 0);
            for (int p = 0; p < 1024 * 1024; p++) {
                //printf("(%f, %f)\n", v.x, v.y);
                SDL_RenderDrawPointF(r, zoom[i] * v.x + 512, zoom[i] * -v.y + 512);
                v = S->apply(v);

                if ((p & (8 * 1024 - 1)) == 0) {
                    SDL_RenderPresent(r);
                }
            }

            SDL_RenderPresent(r);
            SDL_Delay(1000);
        }
        TTF_CloseFont(hFont);
    }
    TTF_Quit();
    SDL_Quit();
    return 0;
}
