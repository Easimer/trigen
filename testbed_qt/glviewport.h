// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: OpenGL viewport declaration
//

#pragma once

#include "common.h"
#include <QWidget>
#include <QOpenGLWidget>
#include "r_renderer.h"
#include "r_queue.h"
#include "arcball_camera.h"

namespace Ui {
class GLViewport;
}

class GLViewport : public QOpenGLWidget
{
    Q_OBJECT

public:
    explicit GLViewport(QWidget *parent = nullptr);

    void set_render_queue_filler(Fun<void(gfx::Render_Queue*)> const& f) {
        fill_render_queue = f;
    }
    
protected:
    void initializeGL() override;
    void resizeGL(int width, int height) override;
    void paintGL() override;

    void mousePressEvent(QMouseEvent* ev) override;
    void mouseReleaseEvent(QMouseEvent* ev) override;
    void mouseMoveEvent(QMouseEvent* ev) override;
    void wheelEvent(QWheelEvent* ev) override;

private:
    Unique_Ptr<Arcball_Camera> camera;
    Unique_Ptr<gfx::IRenderer> renderer;
    Fun<void(gfx::Render_Queue* rq)> fill_render_queue;
};
