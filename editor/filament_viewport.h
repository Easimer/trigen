// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <QApplication>
#include <QWidget>
#include <QResizeEvent>
#include <filament/Engine.h>
#include <filament/SwapChain.h>
#include <filament/Fence.h>
#include <filament/View.h>
#include <filament/Renderer.h>

class Filament_Viewport : public QWidget {
	Q_OBJECT;
public:
	explicit Filament_Viewport(QWidget *parent, filament::Engine::Backend backend) : QWidget(parent) {
		setAttribute(Qt::WA_NativeWindow);
		setAttribute(Qt::WA_PaintOnScreen);
		setAttribute(Qt::WA_NoSystemBackground);

		_engine = filament::Engine::create(backend);
	}

	~Filament_Viewport() {
		_engine->destroy(_view);
		_engine->destroy(_swapChain);
		_engine->destroy(&_engine);
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

		// TODO: recalculate camera matrix

		if (newSize.width() < oldSize.width() || newSize.height() < oldSize.height()) {
			requestRedraw();
		}
	}

	void closeEvent(QCloseEvent *event) override {
		QWidget::closeEvent(event);
		filament::Fence::waitAndDestroy(_engine->createFence());
	}

	void paintEvent(QPaintEvent *event) override {
		requestRedraw();
	}

	bool event(QEvent *event) override {
		switch (event->type()) {
			case QEvent::UpdateRequest:
				if (isVisible()) {
					draw();
				}
				_redrawPending = false;
				return true;
			default:
				return QWidget::event(event);
		}
	}

	void postInit() {
		auto nativeHandle = winId();

		_swapChain = _engine->createSwapChain((void*)nativeHandle);
		_view = _engine->createView();
		_renderer = _engine->createRenderer();
	}

protected:
	void requestRedraw() {
		if (!_redrawPending) {
			_redrawPending = true;
			QApplication::postEvent(this, new QEvent{ QEvent::UpdateRequest });
		}
	}

	void draw() {
		if (_renderer->beginFrame(_swapChain)) {
			_renderer->render(_view);
			_renderer->endFrame();
		}
	}
private:
	filament::Engine *_engine = nullptr;
	filament::SwapChain *_swapChain = nullptr;
	filament::View *_view = nullptr;
	filament::Renderer *_renderer = nullptr;
	bool _redrawPending = false;
};