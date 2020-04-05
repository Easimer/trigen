// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: SDL helper functions
//

#pragma once
#include <SDL.h>

namespace sdl {
    struct Renderer {
        SDL_Window* window;
        SDL_Renderer* renderer;

        Renderer(int w, int h) : Renderer("trigen", w, h) {}

        Renderer(const char* pszTitle, int w, int h) {
            window = SDL_CreateWindow(
                pszTitle,
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

        operator SDL_Window* () const { return window; }
        operator SDL_Renderer* () const { return renderer; }
    };
}