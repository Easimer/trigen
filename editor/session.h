// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: session declaration
//

#pragma once

#include <memory>
#include <arcball_camera.h>
#include <QObject>

#include "filament_wrapper.h"
#include <filament/Scene.h>
#include <math/mat4.h>

class Session : public QObject {
    Q_OBJECT;
public:
    Session(char const *name);

    std::string name() const { return _name; }

public slots:
    void onWindowResize(int w, int h);
    void onMouseDown(int x, int y);
    void onMouseUp(int x, int y);
    void onMouseWheel(int y);
    void onMouseMove(int x, int y);

signals:
    void viewMatrixUpdated(filament::math::mat4f const &mat);

protected:
    void emitViewMatrixUpdated();
private:
    std::string _name;
	std::unique_ptr<Arcball_Camera> _camera;
    filament::Ptr<filament::Scene> _scene;
};