// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: OpenGL viewport declaration
//

#pragma once

#include <QWidget>
#include <QOpenGLWidget>
#include <functional>
#include <memory>
#include <topo.h>
#include "arcball_camera.h"
#include "imgui_impl_qt.h"

namespace topo {
/*
 * An OpenGL viewport widget. It uses the `renderer` library for
 * drawing and integrates the `arcball_camera` library.
 */
class GLViewport : public QOpenGLWidgetImGui {
    Q_OBJECT

public:
    explicit GLViewport(QWidget *parent = nullptr);

protected:
    void
    resizeGL(int width, int height) override;

    void
    mousePressEvent(QMouseEvent *ev) override;
    void
    mouseReleaseEvent(QMouseEvent *ev) override;
    void
    mouseMoveEvent(QMouseEvent *ev) override;
    void
    wheelEvent(QWheelEvent *ev) override;

private:
    std::unique_ptr<Arcball_Camera> camera;
};
}
