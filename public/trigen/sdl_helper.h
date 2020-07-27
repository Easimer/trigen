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
        int const width;
        int const height;

        Renderer(int w, int h) : Renderer("trigen", w, h) {}

        Renderer(const char* pszTitle, int w, int h, Uint32 flags = SDL_WINDOW_SHOWN)
            : width(w), height(h) {
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 5);
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
            SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
            SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
            SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
            SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);

            window = SDL_CreateWindow(
                pszTitle,
                SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                w, h,
                flags
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
