// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: session declaration
//

#pragma once

#include <memory>
#include <arcball_camera.h>
#include <QObject>

#include "world.h"

#include <softbody.h>
#include <r_queue.h>
#include <r_cmd/softbody.h>

class Session : public QObject {
    Q_OBJECT;
public:
    Session(char const *name);

    std::string name() const { return _name; }
    void createPlant(sb::Config const &cfg);
    bool isRunning() const { return _isRunning; }

public slots:
    void onTick(float deltaTime);
    void onRender(gfx::Render_Queue *rq);
    void setRunning(bool isRunning);

private:
    std::string _name;
    Softbody_Render_Parameters _renderParams;
    World _world;
    bool _isRunning = false;
};