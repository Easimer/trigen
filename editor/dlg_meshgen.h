// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: meshgen dialog
//

#pragma once

#include <memory>
#include <QWidget>
#include <r_queue.h>

#include "world_qt.h"

class IDialog_Meshgen {
public:
    virtual ~IDialog_Meshgen() = default;

public slots:
    virtual void onRender(gfx::Render_Queue *rq) = 0;
};

Q_DECLARE_INTERFACE(IDialog_Meshgen, "IDialog_Meshgen");

IDialog_Meshgen *make_meshgen_dialog(QWorld const *world, Entity_Handle entity, QWidget *parent);
