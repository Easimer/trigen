// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: OpenGL viewport implementation
//

#include "imgui_impl_qt.h"
#include <imgui.h>
#include <QOpenGLContext>
#include <QMouseEvent>
#include <QKeyEvent>
#include <QWheelEvent>
#include <Tracy.hpp>
#include "r_queue.h"
#include <ImGuizmo.h>

// TODO:
//  Multiple viewports - multiple contexts:
//    - Allocate new context using ImGui::GetInternalStateSize(),
//      switch it using ImGui::SetInternalState
//    - Expose a method so that client code can make us switch to our
//      state
//    - https://github.com/ocornut/imgui/issues/269
//  Keyboard input:
//    - Don't really need it, because we will only use ImGui for
//      gizmos, which are all mouse-based things

static QOpenGLContext* gpCtx = NULL;
static void* GLGetProcAddress(char const* pFun) {
    assert(gpCtx != NULL);

    if (gpCtx != NULL && pFun != NULL) {
        return (void*)gpCtx->getProcAddress(pFun);
    } else {
        return NULL;
    }
}

QOpenGLWidgetImGui::QOpenGLWidgetImGui(QWidget *parent) : QOpenGLWidget(parent) {
    setMouseTracking(true);
    setFocusPolicy(Qt::StrongFocus);
}

void QOpenGLWidgetImGui::initializeGL() {
    QOpenGLWidget::initializeGL();

    if (!_renderer) {
        assert(context() != NULL);
        gpCtx = context();
        printf("GLViewport context: %p\n", gpCtx);

        auto imguiContext = ImGui::GetCurrentContext();
        assert(imguiContext && "Did you forget to create an ImGui context?");

        _renderer = gfx::make_opengl_renderer(context(), GLGetProcAddress, imguiContext);
        gpCtx = NULL;

        ImGuiIO &io = ImGui::GetIO();
        io.BackendFlags |= ImGuiBackendFlags_HasMouseCursors;
        io.BackendFlags |= ImGuiBackendFlags_HasSetMousePos;
        io.BackendPlatformName = "imgui_impl_qt";
    }
}

void QOpenGLWidgetImGui::resizeGL(int width, int height) {
    QOpenGLWidget::resizeGL(width, height);
    assert(_renderer != NULL);
    assert(context() != NULL);

    unsigned w = width;
    unsigned h = height;
    _renderer->change_resolution(&w, &h);

    ImGuiIO &io = ImGui::GetIO();
    io.DisplaySize.x = width;
    io.DisplaySize.y = height;
}

void QOpenGLWidgetImGui::paintGL() {
    ZoneScoped;
    _renderer->new_frame();

    if (_firstFrame) {
        _firstFrame = false;
        ImGui::NewFrame();
        ImGuizmo::BeginFrame();
    }

    onRender(_renderer.get());
    ImGui::Render();
    _renderer->present();
    ImGui::NewFrame();

    ImGuizmo::BeginFrame();

	ImGuiIO &io = ImGui::GetIO();
	ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);
}

void QOpenGLWidgetImGui::setImGuiMouseButton(Qt::MouseButton button, bool state) {
    ImGuiIO &io = ImGui::GetIO();
    switch (button) {
    case Qt::MouseButton::LeftButton:
        io.MouseDown[0] = state;
        break;
    case Qt::MouseButton::RightButton:
        io.MouseDown[1] = state;
        break;
    case Qt::MouseButton::MiddleButton:
        io.MouseDown[2] = state;
        break;
    }
}

void QOpenGLWidgetImGui::mousePressEvent(QMouseEvent *ev) {
    ImGuiIO &io = ImGui::GetIO();
    if (io.WantCaptureMouse) {
        setImGuiMouseButton(ev->button(), true);
        ev->accept();
    } else {
        ev->ignore();
    }
}

void QOpenGLWidgetImGui::mouseReleaseEvent(QMouseEvent *ev) {
    ImGuiIO &io = ImGui::GetIO();
    if (io.WantCaptureMouse) {
        setImGuiMouseButton(ev->button(), false);
        ev->accept();
    } else {
        ev->ignore();
    }
}

void QOpenGLWidgetImGui::mouseMoveEvent(QMouseEvent *ev) {
    ImGuiIO &io = ImGui::GetIO();
    io.MousePos = ImVec2(ev->x(), ev->y());
    ev->ignore();
}

void QOpenGLWidgetImGui::wheelEvent(QWheelEvent *ev) {
    ImGuiIO &io = ImGui::GetIO();
    if (io.WantCaptureMouse) {
        io.MouseWheel += ev->delta();
        ev->accept();
    } else {
        ev->ignore();
    }
}

void QOpenGLWidgetImGui::keyPressEvent(QKeyEvent *ev) {
    ImGuiIO &io = ImGui::GetIO();
    // TODO(danielm): 
}

void QOpenGLWidgetImGui::keyReleaseEvent(QKeyEvent *ev) {
    ImGuiIO &io = ImGui::GetIO();
    // TODO(danielm): 
}

void QOpenGLWidgetImGui::leaveEvent(QEvent *ev) {
    ImGuiIO &io = ImGui::GetIO();
    io.MousePos = ImVec2(-FLT_MAX, -FLT_MAX);
}
