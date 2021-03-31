// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <QWidget>

#include "renderer.h"

class Filament_Viewport : public QWidget {
	Q_OBJECT;
public:
	explicit Filament_Viewport(QWidget *parent = nullptr);

	QPaintEngine *paintEngine() const override final;
	void resizeEvent(QResizeEvent *resizeEvent) override;
	void closeEvent(QCloseEvent *event) override;
	void paintEvent(QPaintEvent *event) override;
	bool event(QEvent *event) override;
	void setRenderer(Renderer *renderer);
	void mousePressEvent(QMouseEvent *event) override;
	void mouseReleaseEvent(QMouseEvent *event) override;
	void mouseMoveEvent(QMouseEvent *event) override;
	void wheelEvent(QWheelEvent *event) override;

signals:
	void onMouseDown(int x, int y);
	void onMouseUp(int x, int y);
	void onMouseMove(int x, int y);
	void onMouseWheel(int y);
	void onWindowResize(int x, int y);

protected:
	void requestRedraw();
	void requestCameraProjectionUpdate();

private:
	Renderer *_renderer = nullptr;
	bool _redrawPending = false;
};