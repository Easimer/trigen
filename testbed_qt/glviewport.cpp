// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: OpenGL viewport implementation
//

#include "common.h"
#include "glviewport.h"
#include "ui_glviewport.h"
#include "r_queue.h"

#include <QOpenGLContext>
#include <QMouseEvent>
#include <QWheelEvent>

static QOpenGLContext* gpCtx = NULL;
static void* GLGetProcAddress(char const* pFun) {
    assert(gpCtx != NULL);

    if (gpCtx != NULL && pFun != NULL) {
        return gpCtx->getProcAddress(pFun);
    } else {
        return NULL;
    }
}


GLViewport::GLViewport(QWidget *parent) :
    QOpenGLWidget(parent) {
    QSurfaceFormat fmt;
    fmt.setVersion(4, 4);
    fmt.setProfile(QSurfaceFormat::CoreProfile);
    setFormat(fmt);

    camera = create_arcball_camera();
}

void GLViewport::initializeGL() {
    QOpenGLWidget::initializeGL();

    if (!renderer) {
        makeCurrent();
        assert(context() != NULL);
        gpCtx = context();
        renderer = gfx::make_opengl_renderer(context(), GLGetProcAddress);
        gpCtx = NULL;
    }
}

void GLViewport::resizeGL(int width, int height) {
    QOpenGLWidget::resizeGL(width, height);
    assert(renderer != NULL);
    assert(context() != NULL);

    unsigned w = width;
    unsigned h = height;
    renderer->change_resolution(&w, &h);
    camera->set_screen_size(w, h);
}

void GLViewport::paintGL() {
    gfx::Render_Queue rq(4096);
    if (fill_render_queue) {
        fill_render_queue(&rq);
    }

    rq.execute(renderer.get());
}

void GLViewport::mousePressEvent(QMouseEvent* ev) {
    if (ev->button() == Qt::LeftButton) {
        camera->mouse_down(ev->x(), ev->y());
        ev->accept();
    }

    renderer->set_camera(camera->get_view_matrix());
}

void GLViewport::mouseReleaseEvent(QMouseEvent* ev) {
    if (ev->button() == Qt::LeftButton) {
        camera->mouse_up(ev->x(), ev->y());
        ev->accept();
    }

    renderer->set_camera(camera->get_view_matrix());
}

void GLViewport::mouseMoveEvent(QMouseEvent* ev) {
    if (camera->mouse_move(ev->x(), ev->y())) {
        ev->accept();
    }

    renderer->set_camera(camera->get_view_matrix());
}

void GLViewport::wheelEvent(QWheelEvent* ev) {
    camera->mouse_wheel(ev->delta());
    ev->accept();

    renderer->set_camera(camera->get_view_matrix());
}
