// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"
#include "filament_viewport.h"

#include <QApplication>
#include <QResizeEvent>

Filament_Viewport::Filament_Viewport(QWidget *parent) : QWidget(parent) {
	setAttribute(Qt::WA_NativeWindow);
	setAttribute(Qt::WA_PaintOnScreen);
	setAttribute(Qt::WA_NoSystemBackground);
}

QPaintEngine *Filament_Viewport::paintEngine() const {
	// We'll handle drawing
	return nullptr;
}

void Filament_Viewport::resizeEvent(QResizeEvent *resizeEvent) {
	QWidget::resizeEvent(resizeEvent);
	auto oldSize = resizeEvent->oldSize();
	auto newSize = resizeEvent->size();

	if (newSize.width() < 0 || newSize.height() < 0) {
		return;
	}

	requestCameraProjectionUpdate();

	if (newSize.width() < oldSize.width() || newSize.height() < oldSize.height()) {
		requestRedraw();
	}
}

void Filament_Viewport::closeEvent(QCloseEvent *event) {
	QWidget::closeEvent(event);

	if (_renderer) {
		_renderer->onClose();
	}
}

void Filament_Viewport::paintEvent(QPaintEvent *event) {
	requestRedraw();
}

bool Filament_Viewport::event(QEvent *event) {
	switch (event->type()) {
		case QEvent::UpdateRequest:
			if (isVisible() && _renderer) {
				_renderer->draw();
			}
			_redrawPending = false;
			return true;
		default:
			return QWidget::event(event);
	}
}

void Filament_Viewport::setRenderer(Renderer *renderer) {
	_renderer = renderer;
	requestCameraProjectionUpdate();
}

void Filament_Viewport::mousePressEvent(QMouseEvent *event) {
	if (event->button() == Qt::LeftButton) {
		emit onMouseDown(event->x(), event->y());
		event->accept();
	}
}

void Filament_Viewport::mouseReleaseEvent(QMouseEvent *event) {
	if (event->button() == Qt::LeftButton) {
		emit onMouseUp(event->x(), event->y());
		event->accept();
	}
}

void Filament_Viewport::mouseMoveEvent(QMouseEvent *event) {
	emit onMouseMove(event->x(), event->y());
	event->accept();
}

void Filament_Viewport::wheelEvent(QWheelEvent *event) {
	emit onMouseWheel(event->delta());
	event->accept();
}

void Filament_Viewport::requestRedraw() {
	if (!_redrawPending) {
		_redrawPending = true;
		QApplication::postEvent(this, new QEvent{ QEvent::UpdateRequest });
	}
}

void Filament_Viewport::requestCameraProjectionUpdate() {
	if (_renderer) {
		auto const pixelRatio = devicePixelRatio();
		auto w = width() * pixelRatio;
		auto h = height() * pixelRatio;
		_renderer->updateCameraProjection(w, h);
	}
}