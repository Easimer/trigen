// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: OpenGL viewport declaration
//

#pragma once

#include <QWidget>
#include <QOpenGLWidget>
#include <functional>
#include <memory>
#include "r_renderer.h"
#include "r_queue.h"
#include "arcball_camera.h"

/*
 * An OpenGL viewport widget. It uses the `renderer` library for
 * drawing and integrates the `arcball_camera` library.
 */
class GLViewport : public QOpenGLWidget {
    Q_OBJECT

public:
    explicit GLViewport(QWidget *parent = nullptr);

    /*
     * Set the render queue filler callback function.
     * Deprecated; use the `rendering` signal instead.
     */
    [[deprecated]]
    void set_render_queue_filler(std::function<void(gfx::Render_Queue*)> const& f) {
        fill_render_queue = f;
    }

signals:
    void rendering(gfx::Render_Queue *rq);
    
protected:
    void initializeGL() override;
    void resizeGL(int width, int height) override;
    void paintGL() override;

    void mousePressEvent(QMouseEvent* ev) override;
    void mouseReleaseEvent(QMouseEvent* ev) override;
    void mouseMoveEvent(QMouseEvent* ev) override;
    void wheelEvent(QWheelEvent* ev) override;

private:
    std::unique_ptr<Arcball_Camera> camera;
    std::unique_ptr<gfx::IRenderer> renderer;
    std::function<void(gfx::Render_Queue* rq)> fill_render_queue;
};
