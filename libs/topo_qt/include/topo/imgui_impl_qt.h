// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: dear imgui platform backend for Qt
//

#pragma once

#include <memory>

#include <QOpenGLWidget>

#include <topo.h>

namespace topo {
class QOpenGLWidgetImGui : public QOpenGLWidget {
    Q_OBJECT
public:
    QOpenGLWidgetImGui(QWidget *parent = nullptr);

    topo::UPtr<topo::IInstance> &
    renderer() {
        return _instance;
    }

signals:
    void
    rendering(topo::IRender_Queue *rq);

protected:
    void
    initializeGL() override;
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
    void
    keyPressEvent(QKeyEvent *ev) override;
    void
    keyReleaseEvent(QKeyEvent *ev) override;
    void
    leaveEvent(QEvent *ev) override;

private:
    void
    paintGL() override;
    void
    setImGuiMouseButton(Qt::MouseButton button, bool state);

private:
    topo::UPtr<topo::IInstance> _instance;
    bool _firstFrame = true;
};
}
