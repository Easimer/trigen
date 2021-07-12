// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#include <SDL.h>

#include <topo_sdl.h>

#include <imgui.h>
#include <imgui_impl_sdl.h>

namespace topo {

class SDL_Window : public topo::ISDL_Window {
public:
    SDL_Window(
        ::SDL_Window *window,
        SDL_Renderer *renderer,
        UPtr<IInstance> &&instance)
        : _window(window)
        , _renderer(renderer)
        , _instance(std::move(instance)) { }

    ~SDL_Window() override {
        ImGui_ImplSDL2_Shutdown();

        auto glctx = _instance->GLContext();
        if (glctx) {
            SDL_GL_DeleteContext(glctx);
        }

        _instance.reset();

        if (_renderer) {
            SDL_DestroyRenderer(_renderer);
        }

        if (_window) {
            SDL_DestroyWindow(_window);
        }
    }

    bool
    PollEvent(SDL_Event *ev) override {
        auto& io = ImGui::GetIO();
        bool filtered = false;

        do {
            filtered = false;
            if (SDL_PollEvent(ev)) {
                ImGui_ImplSDL2_ProcessEvent(ev);

                if (io.WantCaptureKeyboard) {
                    switch (ev->type) {
                    case SDL_KEYDOWN:
                    case SDL_KEYUP:
                        filtered = true;
                        break;
                    default:
                        break;
                    }
                }
                if (io.WantCaptureMouse) {
                    switch (ev->type) {
                    case SDL_MOUSEMOTION:
                    case SDL_MOUSEBUTTONDOWN:
                    case SDL_MOUSEBUTTONUP:
                        filtered = true;
                        break;
                    }
                }

                switch (ev->type) {
                case SDL_WINDOWEVENT: {
                    switch (ev->window.event) {
                    case SDL_WINDOWEVENT_RESIZED: {
                        _instance->ResolutionChanged(
                            ev->window.data1, ev->window.data2);
                        break;
                    }
                    }
                    break;
                }
                }
            } else {
                return false;
            }
        } while (filtered);

        return true;
    }

    void
    ResolutionChanged(unsigned width, unsigned height) override { }

    ImGuiContext *
        ImguiContext() override {
        return _instance->ImguiContext();
        }

    void
    Present() override {
        ImGui::Render();
        _instance->Present();

        auto const uiTimeEnd = SDL_GetPerformanceCounter();
        auto const flFrameTime = (uiTimeEnd - _timeFrameStart)
            / (double)SDL_GetPerformanceFrequency();

        SDL_GL_SwapWindow(_window);

        _timeFrameStart = SDL_GetPerformanceCounter();

        auto t = 16.6666666 - 1000 * flFrameTime;
        if (t > 1) {
            SDL_Delay(t);
        }
    }

    void
    NewFrame() override {
        _instance->NewFrame();
        ImGui_ImplSDL2_NewFrame(_window);
        ImGui::NewFrame();
    }

    void *
    GLContext() override {
        return _instance->GLContext();
    }

    bool
    CreateTexture(
        Texture_ID *outHandle,
        unsigned width,
        unsigned height,
        Texture_Format format,
        void const *image) override {
        return _instance->CreateTexture(
            outHandle, width, height, format, image);
    }

    void
    DestroyTexture(Texture_ID texture) override {
        _instance->DestroyTexture(texture);
    }

    void
    BeginModelManagement() override {
        _instance->BeginModelManagement();
    }

    void
    FinishModelManagement() override {
        _instance->FinishModelManagement();
    }

    bool
    CreateModel(Model_ID *outHandle, Model_Descriptor const *descriptor)
        override {
        return _instance->CreateModel(outHandle, descriptor);
    }

    void
    DestroyModel(Model_ID model) override {
        _instance->DestroyModel(model);
    }

    bool
    CreateUnlitMaterial(Material_ID *outHandle, Texture_ID diffuse) override {
        return _instance->CreateUnlitMaterial(outHandle, diffuse);
    }

    bool
    CreateUnlitTransparentMaterial(Material_ID *outHandle, Texture_ID diffuse)
        override {
        return _instance->CreateUnlitTransparentMaterial(outHandle, diffuse);
    }

    bool
    CreateLitMaterial(
        Material_ID *outHandle,
        Texture_ID diffuse,
        Texture_ID normal) override {
        return _instance->CreateLitMaterial(outHandle, diffuse, normal);
    }

    bool
    CreateSolidColorMaterial(Material_ID *outHandle, glm::vec3 const &color)
        override {
        return _instance->CreateSolidColorMaterial(outHandle, color);
    }

    void
    DestroyMaterial(Material_ID material) override {
        _instance->DestroyMaterial(material);
    }

    bool
    CreateRenderable(
        Renderable_ID *outHandle,
        Model_ID model,
        Material_ID material) override {
        return _instance->CreateRenderable(outHandle, model, material);
    }

    void
    DestroyRenderable(Renderable_ID renderable) override {
        _instance->DestroyRenderable(renderable);
    }

    IRender_Queue *
    BeginRendering() override {
        return _instance->BeginRendering();
    }

    void
    FinishRendering() override {
        _instance->FinishRendering();
    }

    void
    SetEyeCamera(Transform const &transform) override {
        _instance->SetEyeCamera(transform);
    }

    void
    SetEyeViewMatrix(glm::mat4 const &matView) override {
        _instance->SetEyeViewMatrix(matView);
    }

    bool
    CreateRenderableLinesStreaming(
        Renderable_ID *outHandle,
        glm::vec3 const *endpoints,
        size_t lineCount,
        glm::vec3 const &colorBegin,
        glm::vec3 const &colorEnd) override {
        return _instance->CreateRenderableLinesStreaming(
            outHandle, endpoints, lineCount, colorBegin, colorEnd);
    }

private:
    ::SDL_Window *_window;
    SDL_Renderer *_renderer;
    UPtr<IInstance> _instance;

    decltype(SDL_GetPerformanceCounter()) _timeFrameStart;
};

UPtr<ISDL_Window>
MakeWindow(Surface_Config const &cfg) {
    SDL_Init(SDL_INIT_EVERYTHING);

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 4);
    SDL_GL_SetAttribute(
        SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
#ifndef NDEBUG
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_DEBUG_FLAG);
#endif

    auto window = SDL_CreateWindow(
        cfg.title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, cfg.width,
        cfg.height,
        SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);

    if (window == nullptr) {
        fprintf(stderr, "[ topo ] Failed to open an SDL window: %s\n", SDL_GetError());
        return nullptr;
    }

    auto renderer = SDL_CreateRenderer(
        window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

    if (renderer == nullptr) {
        fprintf(stderr, "[ topo ] Failed to create an SDL renderer: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        return nullptr;
    }

    auto glctx = SDL_GL_CreateContext(window);

    if (glctx == nullptr) {
        fprintf(stderr, "[ topo ] Failed to create a GL context: %s\n", SDL_GetError());
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        return nullptr;
    }

    SDL_GL_SetSwapInterval(0);

    IMGUI_CHECKVERSION();
    auto imguiCtx = ImGui::CreateContext();

    ImGui::GetIO();
    ImGui::StyleColorsDark();
    ImGui_ImplSDL2_InitForOpenGL(window, glctx);

    auto topoInstance
        = topo::MakeInstance(glctx, SDL_GL_GetProcAddress, imguiCtx);

    if (topoInstance == nullptr) {
        fprintf(stderr, "[ topo ] Failed to create topo::IInstance\n");
        SDL_GL_DeleteContext(glctx);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        return nullptr;
    }

    topoInstance->ResolutionChanged(cfg.width, cfg.height);

    return std::make_unique<topo::SDL_Window>(window, renderer, std::move(topoInstance));
}

}
