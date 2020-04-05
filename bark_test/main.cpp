// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: Bark texture generation test
//

#include <SDL.h>
#include <cmath>
#include <ctime>
#include <cstdio>

static float randf() {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

struct Brown_Noise_Generator {
    float current;

    Brown_Noise_Generator() : current(randf()) {}

    float next() {
        float integral;
        current = modf(current + randf(), &integral);
        return current;
    }
};

struct Bark_Texture {
    float N, R;
    Brown_Noise_Generator bng;

    Bark_Texture() : N(16.0), R(0.2) {}

    SDL_Color Sample(float x, float y) {
        float val = saw(N * (x + R * bng.next()));
        Uint8 byte = (Uint8)(val * 127);
        return {byte, byte, byte, 255};
    }

    float saw(float t) {
        float integral;
        auto f = modf(t, &integral);
        if (f > 0.5) {
            return 2 * f;
        } else {
            return 2 * (1 - f);
        }
    }
};

struct Renderer {
    SDL_Window* window;
    SDL_Renderer* renderer;

    Renderer(int w, int h) {
        window = SDL_CreateWindow(
            "Bark Test",
            SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
            w, h,
            SDL_WINDOW_SHOWN
        );
        if (window != NULL) {
            renderer = SDL_CreateRenderer(
                window, -1,
                SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC
            );
        }
    }

    ~Renderer() {
        if (renderer) {
            SDL_DestroyRenderer(renderer);
        }
        if (window) {
            SDL_DestroyWindow(window);
        }
    }

    operator bool() const { return window != NULL && renderer != NULL; }
};

#define TEX_SIZE (512)
#define TEX_SIZE_F ((float)TEX_SIZE)

int main(int argc, char** argv) {
    SDL_Init(SDL_INIT_VIDEO);
    Renderer renderer(TEX_SIZE, TEX_SIZE);
    if (renderer) {
        srand(time(NULL));
        Bark_Texture tex;

        printf("Bark bump map generation demo\n");
        printf("Real Time Design and Animation of Fractal Plants and Trees by Peter E. Oppenheimer\n\n");
        printf("Press Up-Down to change the number of ridges\n");
        printf("Press Left-Right to change the roughness of the bark\n");
        printf("Press Escape to quit\n");

        bool exit = false;
        while (!exit) {
            SDL_Event ev;
            while (SDL_PollEvent(&ev)) {
                switch (ev.type) {
                case SDL_KEYUP:
                    switch (ev.key.keysym.sym) {
                    case SDLK_ESCAPE:
                        exit = true;
                        break;
                    case SDLK_UP:
                        tex.N += 1.0f;
                        if (tex.N > 128) tex.N = 128;
                        break;
                    case SDLK_DOWN:
                        tex.N -= 1.0f;
                        if (tex.N < 1) tex.N = 1;
                        break;
                    case SDLK_LEFT:
                        tex.R -= 0.05f;
                        if (tex.R < 0.0f) tex.R = 0.0f;
                        break;
                    case SDLK_RIGHT:
                        tex.R += 0.05f;
                        if (tex.R > 1.0f) tex.R = 1.0f;
                        break;
                    }

                    printf("N=%f R=%f\n", tex.N, tex.R);
                    break;
                case SDL_QUIT:
                    exit = true;
                    break;
                }
            }

            SDL_RenderClear(renderer.renderer);
            for (int y = 0; y < TEX_SIZE; y++) {
                for (int x = 0; x < TEX_SIZE; x++) {
                    auto color = tex.Sample(x / TEX_SIZE_F, y / TEX_SIZE_F);
                    SDL_SetRenderDrawColor(renderer.renderer, color.r, color.g, color.b, color.a);
                    SDL_RenderDrawPoint(renderer.renderer, x, y);
                }
            }
            SDL_RenderPresent(renderer.renderer);
        }
    }

    SDL_Quit();
    return 0;
}
