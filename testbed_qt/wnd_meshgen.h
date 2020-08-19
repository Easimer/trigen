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

#include <trigen/meshbuilder.h>

class Window_Meshgen : public QDialog {
    Q_OBJECT;
public:
    Window_Meshgen(
        sb::Unique_Ptr<sb::ISoftbody_Simulation>& simulation,
        QMainWindow* parent = nullptr
    );

public slots:
    void render(gfx::Render_Queue* rq);
    void update_mesh();

private:
    QHBoxLayout layout;
    sb::Unique_Ptr<sb::ISoftbody_Simulation>& simulation;
    Mesh_Builder::Optimized_Mesh mesh;
};
