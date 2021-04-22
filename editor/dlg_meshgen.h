// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: meshgen dialog
//

#pragma once

#include <memory>
#include <QDialog>
#include <r_queue.h>

#include "world_qt.h"

class Base_Dialog_Meshgen : public QDialog {
    Q_OBJECT;

public:
    Base_Dialog_Meshgen(QWidget *parent = nullptr)
        : QDialog(parent) {
    }

    virtual ~Base_Dialog_Meshgen() = default;

public slots:
    virtual void onRender(gfx::Render_Queue *rq) = 0;
};

Base_Dialog_Meshgen *make_meshgen_dialog(QWorld const *world, Entity_Handle entity, QWidget *parent);
