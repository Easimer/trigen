// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: OpenGL viewport implementation
//

#include <topo/glviewport.h>

#include <QMouseEvent>
#include <QOpenGLContext>
#include <QWheelEvent>

#include <Tracy.hpp>

static QOpenGLContext* gpCtx = NULL;
static void* GLGetProcAddress(char const* pFun) {
    assert(gpCtx != NULL);

    if (gpCtx != NULL && pFun != NULL) {
        return (void*)gpCtx->getProcAddress(pFun);
    } else {
        return NULL;
    }
}


namespace topo {
GLViewport::GLViewport(QWidget *parent)
    : QOpenGLWidgetImGui(parent) {
    camera = create_arcball_camera();
}

void
GLViewport::resizeGL(int width, int height) {
    QOpenGLWidgetImGui::resizeGL(width, height);

    unsigned w = width;
    unsigned h = height;
    camera->set_screen_size(w, h);
}

void
GLViewport::mousePressEvent(QMouseEvent *ev) {
    QOpenGLWidgetImGui::mousePressEvent(ev);
    if (ev->isAccepted()) {
        return;
    }

    if (ev->button() == Qt::LeftButton) {
        camera->mouse_down(ev->x(), ev->y());
        ev->accept();
    }
}

void
GLViewport::mouseReleaseEvent(QMouseEvent *ev) {
    QOpenGLWidgetImGui::mouseReleaseEvent(ev);
    if (ev->isAccepted()) {
        return;
    }

    if (ev->button() == Qt::LeftButton) {
        camera->mouse_up(ev->x(), ev->y());
        ev->accept();
    }
}

void
GLViewport::mouseMoveEvent(QMouseEvent *ev) {
    QOpenGLWidgetImGui::mouseMoveEvent(ev);
    if (ev->isAccepted()) {
        return;
    }

    if (camera->mouse_move(ev->x(), ev->y())) {
        ev->accept();
    }
}

void
GLViewport::wheelEvent(QWheelEvent *ev) {
    QOpenGLWidgetImGui::wheelEvent(ev);
    if (ev->isAccepted()) {
        return;
    }

    camera->mouse_wheel(ev->delta());
    ev->accept();
}
}
