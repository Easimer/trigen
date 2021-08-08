// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: OpenGL viewport implementation
//

#include <topo/imgui_impl_qt.h>

#include <ImGuizmo.h>
#include <imgui.h>

#include <QKeyEvent>
#include <QMouseEvent>
#include <QOpenGLContext>
#include <QWheelEvent>

#include <Tracy.hpp>

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

namespace topo {
static QOpenGLContext *gpCtx = NULL;
static void *
GLGetProcAddress(char const *pFun) {
    assert(gpCtx != NULL);

    if (gpCtx != NULL && pFun != NULL) {
        return (void *)gpCtx->getProcAddress(pFun);
    } else {
        return NULL;
    }
}

QOpenGLWidgetImGui::QOpenGLWidgetImGui(QWidget *parent)
    : QOpenGLWidget(parent) {
    setMouseTracking(true);
    setFocusPolicy(Qt::StrongFocus);
}

void
QOpenGLWidgetImGui::initializeGL() {
    QOpenGLWidget::initializeGL();

    if (!_instance) {
        assert(context() != NULL);
        gpCtx = context();

        auto imguiContext = ImGui::GetCurrentContext();
        assert(imguiContext && "Did you forget to create an ImGui context?");

        _instance
            = topo::MakeInstance(context(), GLGetProcAddress, imguiContext);
        gpCtx = NULL;

        ImGuiIO &io = ImGui::GetIO();
        io.BackendFlags |= ImGuiBackendFlags_HasMouseCursors;
        io.BackendFlags |= ImGuiBackendFlags_HasSetMousePos;
        io.BackendPlatformName = "imgui_impl_qt";
        io.FontDefault = io.Fonts->AddFontDefault();
    }
}

void
QOpenGLWidgetImGui::resizeGL(int width, int height) {
    QOpenGLWidget::resizeGL(width, height);
    assert(_instance != NULL);
    assert(context() != NULL);

    if (width > 0 && height > 0) {
        _instance->ResolutionChanged(width, height);
    } else {
        _instance->ResolutionChanged(0, 0);
    }

    ImGuiIO &io = ImGui::GetIO();
    io.DisplaySize.x = width;
    io.DisplaySize.y = height;
}

void
QOpenGLWidgetImGui::paintGL() {
    ZoneScoped;

    _instance->NewFrame();
    ImGui::NewFrame();
    ImGuiIO &io = ImGui::GetIO();

    auto *rq = _instance->BeginRendering();

    if (_firstFrame) {
        _firstFrame = false;
        ImGuizmo::BeginFrame();
    }

    emit rendering(rq);
    _instance->FinishRendering();

    ImGuizmo::BeginFrame();

    ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);
}

void
QOpenGLWidgetImGui::setImGuiMouseButton(Qt::MouseButton button, bool state) {
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

void
QOpenGLWidgetImGui::mousePressEvent(QMouseEvent *ev) {
    ImGuiIO &io = ImGui::GetIO();
    if (io.WantCaptureMouse) {
        setImGuiMouseButton(ev->button(), true);
        ev->accept();
    } else {
        ev->ignore();
    }
}

void
QOpenGLWidgetImGui::mouseReleaseEvent(QMouseEvent *ev) {
    ImGuiIO &io = ImGui::GetIO();
    if (io.WantCaptureMouse) {
        setImGuiMouseButton(ev->button(), false);
        ev->accept();
    } else {
        ev->ignore();
    }
}

void
QOpenGLWidgetImGui::mouseMoveEvent(QMouseEvent *ev) {
    ImGuiIO &io = ImGui::GetIO();
    io.MousePos = ImVec2(ev->x(), ev->y());
    ev->ignore();
}

void
QOpenGLWidgetImGui::wheelEvent(QWheelEvent *ev) {
    ImGuiIO &io = ImGui::GetIO();
    if (io.WantCaptureMouse) {
        io.MouseWheel += ev->delta();
        ev->accept();
    } else {
        ev->ignore();
    }
}

void
QOpenGLWidgetImGui::keyPressEvent(QKeyEvent *ev) {
    ImGuiIO &io = ImGui::GetIO();
    // TODO(danielm):
}

void
QOpenGLWidgetImGui::keyReleaseEvent(QKeyEvent *ev) {
    ImGuiIO &io = ImGui::GetIO();
    // TODO(danielm):
}

void
QOpenGLWidgetImGui::leaveEvent(QEvent *ev) {
    ImGuiIO &io = ImGui::GetIO();
    io.MousePos = ImVec2(-FLT_MAX, -FLT_MAX);
}
}
