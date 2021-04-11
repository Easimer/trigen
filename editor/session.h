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

#include "world.h"
#include "filament_factory.h"

#include <softbody.h>

class Session : public QObject {
    Q_OBJECT;
public:
    Session(Filament_Factory *factory, char const *name);

    std::string name() const { return _name; }
    void createPlant(sb::Config const &cfg);
    filament::Scene *scene();

public slots:
    void onWindowResize(int w, int h);
    void onMouseDown(int x, int y);
    void onMouseUp(int x, int y);
    void onMouseWheel(int y);
    void onMouseMove(int x, int y);
    void onTick(float deltaTime);

signals:
    void cameraUpdated(filament::math::float3 const &eye, filament::math::float3 const &center);

protected:
    void emitCameraUpdated();
private:
    Filament_Factory *_factory;
    std::string _name;
	std::unique_ptr<Arcball_Camera> _camera;
    filament::Engine *_engine;
    filament::Ptr<filament::Scene> _scene;
    World _world;
};