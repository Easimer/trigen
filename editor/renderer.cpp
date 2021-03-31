// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: renderer
//

#include "stdafx.h"
#include "renderer.h"

#include <filament/Camera.h>
#include <filament/Engine.h>
#include <filament/IndexBuffer.h>
#include <filament/Material.h>
#include <filament/MaterialInstance.h>
#include <filament/RenderableManager.h>
#include <filament/Scene.h>
#include <filament/Skybox.h>
#include <filament/TransformManager.h>
#include <filament/VertexBuffer.h>
#include <filament/View.h>
#include <filament/SwapChain.h>
#include <filament/Fence.h>
#include <filament/Renderer.h>
#include <filament/Viewport.h>
#include <filament/VertexBuffer.h>
#include <filament/IndexBuffer.h>
#include <utils/Entity.h>
#include <utils/EntityManager.h>

// TODO: 
// - Convert PSP mesh to filament renderable
// - Find out what's wrong with materials

struct Vertex {
    filament::math::float2 position;
    uint32_t color;
};

static const Vertex TRIANGLE_VERTICES[3] = {
    {{ -0.5f, -0.5f }, 0xffff0000u},
    {{ 0.5f, -0.5f },  0xff00ff00u},
    {{ 0.0f,  0.5f },  0xff0000ffu},
};

static constexpr uint16_t TRIANGLE_INDICES[3] = { 0, 1, 2 };

extern "C" {
    extern unsigned long long bakedColor_matc_len;
    extern char const *bakedColor_matc;
}

Renderer::Renderer(filament::Engine::Backend backend, void *nativeHandle) {
    using namespace filament;
    _engine = Engine::create(backend);

    _swapChain = _engine->createSwapChain(nativeHandle);
    _view = _engine->createView();
    _renderer = _engine->createRenderer();
    _camera = _engine->createCamera();

    _view->setCamera(_camera);

    _skybox = Skybox::Builder().color({ 0.1, 0.125, 0.25, 1.0 }).build(*_engine);
    _scene = _engine->createScene();
    _scene->setSkybox(_skybox);
    _view->setScene(_scene);
    _view->setPostProcessingEnabled(false);

    _vb = VertexBuffer::Builder()
        .vertexCount(3)
        .bufferCount(1)
        .attribute(VertexAttribute::POSITION, 0, VertexBuffer::AttributeType::FLOAT2, 0, 12)
        .attribute(VertexAttribute::COLOR, 0, VertexBuffer::AttributeType::UBYTE4, 8, 12)
        .normalized(VertexAttribute::COLOR)
        .build(*_engine);
    _vb->setBufferAt(*_engine, 0,
        VertexBuffer::BufferDescriptor(TRIANGLE_VERTICES, 36, nullptr));
    _ib = IndexBuffer::Builder()
        .indexCount(3)
        .bufferType(IndexBuffer::IndexType::USHORT)
        .build(*_engine);
    _ib->setBuffer(*_engine,
        IndexBuffer::BufferDescriptor(TRIANGLE_INDICES, 6, nullptr));
    _mat = Material::Builder()
        .package(bakedColor_matc, bakedColor_matc_len)
        .build(*_engine);
    _renderable = utils::EntityManager::get().create();
    RenderableManager::Builder(1)
        .boundingBox({ { -1, -1, -1 }, { 1, 1, 1 } })
        .geometry(0, RenderableManager::PrimitiveType::TRIANGLES, _vb, _ib, 0, 3)
        .culling(false)
        .receiveShadows(false)
        .castShadows(false)
        .build(*_engine, _renderable);
    _scene->addEntity(_renderable);
}

void Renderer::onClose() {
    filament::Fence::waitAndDestroy(_engine->createFence());
}

void Renderer::draw() {
    if (_renderer->beginFrame(_swapChain)) {
        _renderer->render(_view);
        _renderer->endFrame();
    }
}

void Renderer::updateCameraProjection(uint32_t w, uint32_t h) {
    _view->setViewport({ 0, 0, w, h });
}