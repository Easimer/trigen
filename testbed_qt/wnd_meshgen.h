// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: meshgen window declaration
//

#pragma once

#include "common.h"
#include <memory>
#include <QWindow>
#include <QLayout>
#include "softbody.h"
#include "glviewport.h"

class Window_Meshgen : public QDialog {
    Q_OBJECT;
public:
    Window_Meshgen(
        sb::Unique_Ptr<sb::ISoftbody_Simulation>& simulation,
        QMainWindow* parent = nullptr
    );

public slots:
    void render(gfx::Render_Queue* rq);

private:
    QHBoxLayout layout;
    sb::Unique_Ptr<sb::ISoftbody_Simulation>& simulation;
};
