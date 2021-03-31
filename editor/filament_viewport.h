// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <QApplication>
#include <QWidget>
#include <QResizeEvent>

#include "renderer.h"

class Filament_Viewport : public QWidget {
	Q_OBJECT;
public:
	explicit Filament_Viewport(QWidget *parent) : QWidget(parent) {
		setAttribute(Qt::WA_NativeWindow);
		setAttribute(Qt::WA_PaintOnScreen);
		setAttribute(Qt::WA_NoSystemBackground);
	}

	QPaintEngine *paintEngine() const override final {
		// We'll handle drawing
		return nullptr;
	}

	void resizeEvent(QResizeEvent *resizeEvent) override {
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

	void closeEvent(QCloseEvent *event) override {
		QWidget::closeEvent(event);

		if (_renderer) {
			_renderer->onClose();
		}
	}

	void paintEvent(QPaintEvent *event) override {
		requestRedraw();
	}

	bool event(QEvent *event) override {
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

	void setRenderer(Renderer *renderer) {
		_renderer = renderer;
		requestCameraProjectionUpdate();
	}

protected:
	void requestRedraw() {
		if (!_redrawPending) {
			_redrawPending = true;
			QApplication::postEvent(this, new QEvent{ QEvent::UpdateRequest });
		}
	}

	void requestCameraProjectionUpdate() {
		if (_renderer) {
			auto const pixelRatio = devicePixelRatio();
			auto w = width() * pixelRatio;
			auto h = height() * pixelRatio;
			_renderer->updateCameraProjection(w, h);
		}
	}
private:
	Renderer *_renderer = nullptr;
	bool _redrawPending = false;
};