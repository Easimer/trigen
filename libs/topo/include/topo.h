// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#pragma once

#if _WIN32
#if defined(TOPO_BUILDING)
#define TOPO_EXPORT __declspec(dllexport)
#else
#define TOPO_EXPORT __declspec(dllimport)
#endif
#else
#define TOPO_EXPORT
#endif

#include <memory>

#include <glm/vec3.hpp>
#include <glm/gtc/quaternion.hpp>

#include <imgui.h>

namespace topo {

using Model_ID = void *;
using Texture_ID = void *;
using Material_ID = void *;
using Renderable_ID = void *;

template <typename T> using UPtr = std::unique_ptr<T>;

struct Model_Descriptor {
    size_t vertex_count = 0;
    void const *vertices = nullptr;
    void const *normals = nullptr;
    void const *uv = nullptr;
    size_t element_count = 0;
    unsigned const *elements = nullptr;
};

enum class Texture_Format {
    RGB888,
    SRGB888,
    RGBA8888,
};

struct Transform {
    glm::vec3 position = { 0, 0, 0 };
    glm::quat rotation = { 1, 0, 0, 0 };
    glm::vec3 scale = { 1, 1, 1 };
};

class TOPO_EXPORT IRender_Queue {
public:
    virtual ~IRender_Queue() = default;

    virtual void
    Submit(Renderable_ID renderable, Transform const &transform)
        = 0;

    virtual void
    AddLight(
        glm::vec4 const &color,
        Transform const &transform,
        bool castsShadows = false)
        = 0;
};

class TOPO_EXPORT IInstance {
public:
    virtual ~IInstance() = default;

    virtual void *
    GLContext()
        = 0;

    virtual ImGuiContext *
    ImguiContext()
        = 0;

    virtual void
    NewFrame()
        = 0;

    virtual void
    Present()
        = 0;

    virtual void
    ResolutionChanged(unsigned width, unsigned height)
        = 0;

    virtual bool 
    CreateTexture(Texture_ID *outHandle, unsigned width, unsigned height, Texture_Format format, void const *image)
        = 0;

    virtual void
    DestroyTexture(Texture_ID texture)
        = 0;

    virtual void
    BeginModelManagement()
        = 0;

    virtual void
    FinishModelManagement()
        = 0;

    virtual bool
    CreateModel(Model_ID *outHandle, Model_Descriptor const *descriptor)
        = 0;

    virtual void
    DestroyModel(Model_ID model)
        = 0;

    virtual bool
    CreateUnlitMaterial(
        Material_ID *outHandle,
        Texture_ID diffuse)
        = 0;

    virtual bool
    CreateUnlitTransparentMaterial(
        Material_ID *outHandle,
        Texture_ID diffuse)
        = 0;

    virtual bool
    CreateLitMaterial(
        Material_ID *outHandle,
        Texture_ID diffuse,
        Texture_ID normal)
        = 0;

    virtual bool
    CreateSolidColorMaterial(Material_ID *outHandle, glm::vec3 const &color)
        = 0;

    virtual void
    DestroyMaterial(Material_ID material)
        = 0;

    virtual bool
    CreateRenderable(
        Renderable_ID *outHandle,
        Model_ID model,
        Material_ID material)
        = 0;

    virtual bool
    CreateRenderableLinesStreaming(
        Renderable_ID *outHandle,
        glm::vec3 const *endpoints,
        size_t lineCount,
        glm::vec3 const &colorBegin,
        glm::vec3 const &colorEnd)
        = 0;

    virtual void
    DestroyRenderable(Renderable_ID renderable)
        = 0;

    virtual IRender_Queue *
    BeginRendering()
        = 0;

    virtual void
    FinishRendering()
        = 0;

    virtual void
    SetEyeCamera(Transform const &transform)
        = 0;

    virtual void
    SetEyeViewMatrix(glm::mat4 const &matView)
        = 0;
};

TOPO_EXPORT
UPtr<IInstance>
MakeInstance(
    void *glctx,
    void *(*getProcAddress)(char const *),
    void *imguiContext);

}
