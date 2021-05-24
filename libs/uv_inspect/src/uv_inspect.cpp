// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: UV inspector implementation
//

#include <uv_inspect.hpp>

#include <cstdio>

#include <SDL.h>

#define WND_SIZ (1024)

static void draw_triangle(SDL_Renderer *renderer, glm::vec2 const *points) {
    SDL_FPoint sdl_points[4];

    sdl_points[0] = { points[0].x * WND_SIZ, points[0].y * WND_SIZ };
    sdl_points[1] = { points[1].x * WND_SIZ, points[1].y * WND_SIZ };
    sdl_points[2] = { points[2].x * WND_SIZ, points[2].y * WND_SIZ };
    sdl_points[3] = { points[0].x * WND_SIZ, points[0].y * WND_SIZ };

    SDL_RenderDrawLinesF(renderer, sdl_points, 4);
}

namespace uv_inspector {
    int inspect(glm::vec2 const *texCoords, int count) {
        int ret = 0;
        if (SDL_InitSubSystem(SDL_INIT_VIDEO) != 0) {
            fprintf(stderr, "uv_inspector: SDL_InitSubSystem has failed: %s\n", SDL_GetError());
            goto over;
        }

        auto wnd = ::SDL_CreateWindow("UV inspector", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WND_SIZ, WND_SIZ, SDL_WINDOW_SHOWN);
        if (wnd == nullptr) {
            fprintf(stderr, "uv_inspector: SDL_CreateWindow has failed: %s\n", SDL_GetError());
            goto sdl_shutdown;
        }

        auto renderer = SDL_CreateRenderer(wnd, -1, 0);
        if (renderer == nullptr) {
            fprintf(stderr, "uv_inspector: SDL_CreateRenderer has failed: %s\n", SDL_GetError());
            goto sdl_freewnd;
        }

        ret = 1;
        // Draw triangles
        SDL_SetRenderDrawColor(renderer, 212, 212, 212, SDL_ALPHA_OPAQUE);
        for (size_t i = 0; i < count; i += 3) {
            draw_triangle(renderer, &texCoords[i]);
        }
        SDL_RenderPresent(renderer);

        SDL_Event ev;
        bool quit = false;
        while (!quit) {
            while (SDL_PollEvent(&ev)) {
                if (ev.type == SDL_QUIT) {
                    quit = true;
                }
            }
        }

    sdl_freerenderer:
        SDL_DestroyRenderer(renderer);
    sdl_freewnd:
        SDL_DestroyWindow(wnd);
    sdl_shutdown:
        SDL_QuitSubSystem(SDL_INIT_VIDEO);
    over:
        return ret;
    }
}
