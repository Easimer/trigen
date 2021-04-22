// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: meshgen dialog
//

#pragma once

#include <memory>
#include <QWidget>
#include <r_queue.h>

class IDialog_Meshgen {
public:
    virtual ~IDialog_Meshgen() = default;

public slots:
    virtual void onRender(gfx::Render_Queue *rq) = 0;
};

Q_DECLARE_INTERFACE(IDialog_Meshgen, "IDialog_Meshgen");

std::unique_ptr<IDialog_Meshgen> make_meshgen_dialog(QWidget *parent);
