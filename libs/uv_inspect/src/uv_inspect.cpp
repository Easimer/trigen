// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: UV inspector implementation
//

#include <uv_inspect.hpp>

#include <cstdio>

#include <glm/glm.hpp>

#include <SDL.h>
#include <SDL_ttf.h>

#define WND_SIZ (1024)

static void draw_triangle(SDL_Renderer *renderer, glm::vec2 const *points, unsigned const *indices) {
    SDL_FPoint sdl_points[4];

    sdl_points[0] = { points[indices[0]].x * WND_SIZ, points[indices[0]].y * WND_SIZ };
    sdl_points[1] = { points[indices[1]].x * WND_SIZ, points[indices[1]].y * WND_SIZ };
    sdl_points[2] = { points[indices[2]].x * WND_SIZ, points[indices[2]].y * WND_SIZ };
    sdl_points[0] = { points[indices[0]].x * WND_SIZ, points[indices[0]].y * WND_SIZ };

    SDL_RenderDrawLinesF(renderer, sdl_points, 4);
}

static TTF_Font *load_font() {
    TTF_Font *ret = nullptr;
    SDL_RWops *ops = nullptr;

    ops = SDL_RWFromFile("iosevka-regular.ttf", "rb");
    if (ops == nullptr) {
        return nullptr;
    }

    ret = TTF_OpenFontRW(ops, 1, 16);

    return ret;
}

static float area_of_triangle(glm::vec3 const &A, glm::vec3 const &B, glm::vec3 const &C) {
    auto AB = B - A;
    auto AC = C - A;
    return length(cross(AB, AC)) / 2;
}

static float area_of_triangle(glm::vec2 const &A, glm::vec2 const &B, glm::vec2 const &C) {
    return area_of_triangle(glm::vec3{ A, 0 }, { B, 0 }, { C, 0 });
}

static float calculate_space_usage(glm::vec2 const *texCoords, int count) {
    float total = 0;

    for (int i = 0; i < count; i += 3) {
        total += area_of_triangle(texCoords[i + 0], texCoords[i + 1], texCoords[i + 2]);
    }

    return total;
}

namespace uv_inspector {
    int inspect(glm::vec2 const *texCoords, unsigned const *indices, int count) {
        int ret = 0;
        SDL_Event ev;
        bool quit = false;
        SDL_Window *wnd;
        SDL_Renderer *renderer;
        TTF_Font *font;
        size_t i;
        float spaceUsage;
        char textBuffer[128];
        SDL_Texture *texText;
        SDL_Surface *surfText;
        SDL_Rect rectText;

        if (SDL_InitSubSystem(SDL_INIT_VIDEO) != 0) {
            fprintf(stderr, "uv_inspector: SDL_InitSubSystem has failed: %s\n", SDL_GetError());
            goto over;
        }

        wnd = ::SDL_CreateWindow("UV inspector", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WND_SIZ, WND_SIZ, SDL_WINDOW_SHOWN);
        if (wnd == nullptr) {
            fprintf(stderr, "uv_inspector: SDL_CreateWindow has failed: %s\n", SDL_GetError());
            goto sdl_shutdown;
        }

        renderer = SDL_CreateRenderer(wnd, -1, 0);
        if (renderer == nullptr) {
            fprintf(stderr, "uv_inspector: SDL_CreateRenderer has failed: %s\n", SDL_GetError());
            goto sdl_freewnd;
        }

        if (TTF_Init() != 0) {
            fprintf(stderr, "uv_inspector: SDL_CreateRenderer has failed: %s\n", SDL_GetError());
            goto sdl_freerenderer;
        }

        ret = 1;

        SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);
        SDL_RenderClear(renderer);

        font = load_font();
        if (font != nullptr) {
            spaceUsage = calculate_space_usage(texCoords, count);
            snprintf(textBuffer, 127, "Space usage: %f%%", spaceUsage * 100);

            surfText = TTF_RenderUTF8_Blended(font, textBuffer, { 0, 0, 0 });
            texText = SDL_CreateTextureFromSurface(renderer, surfText);
            rectText.w = surfText->w;
            rectText.h = surfText->h;
            rectText.x = WND_SIZ - rectText.w;
            rectText.y = WND_SIZ - rectText.h;
            SDL_FreeSurface(surfText);
            SDL_RenderCopy(renderer, texText, NULL, &rectText);
            TTF_CloseFont(font);
        }

        // Draw triangles
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
        for (i = 0; i < count; i += 3) {
            draw_triangle(renderer, texCoords, &indices[i]);
        }
        SDL_RenderPresent(renderer);

        while (!quit) {
            while (SDL_PollEvent(&ev)) {
                if (ev.type == SDL_QUIT) {
                    quit = true;
                }
            }
        }

        TTF_Quit();
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
