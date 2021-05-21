// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: dear imgui platform backend for Qt
//

#pragma once

#include <memory>
#include <QOpenGLWidget>
#include <r_renderer.h>

class QOpenGLWidgetImGui : public QOpenGLWidget {
public:
    QOpenGLWidgetImGui(QWidget *parent = nullptr);
protected:
    void initializeGL() override;
    void resizeGL(int width, int height) override;

    void mousePressEvent(QMouseEvent* ev) override;
    void mouseReleaseEvent(QMouseEvent* ev) override;
    void mouseMoveEvent(QMouseEvent* ev) override;
    void wheelEvent(QWheelEvent* ev) override;
    void keyPressEvent(QKeyEvent *ev) override;
    void keyReleaseEvent(QKeyEvent *ev) override;
    void leaveEvent(QEvent *ev) override;

    virtual void onRender(gfx::IRenderer *renderer) = 0;

private:
    void paintGL() override;
    void setImGuiMouseButton(Qt::MouseButton button, bool state);

private:
    std::unique_ptr<gfx::IRenderer> _renderer;
    bool _firstFrame = true;
};
