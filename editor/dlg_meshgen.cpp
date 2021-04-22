// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"

#include "dlg_meshgen.h"
#include <QDialog>
#include <marching_cubes.h>
#include <psp/psp.h>

class Dialog_Meshgen : public QDialog, public IDialog_Meshgen {
    Q_OBJECT;
    Q_INTERFACES(IDialog_Meshgen);

public:
    Dialog_Meshgen(QWidget *parent)
        : QDialog(parent) {
    }

    ~Dialog_Meshgen() override = default;

    void onRender(gfx::Render_Queue *rq) override {
    }

private:
};

std::unique_ptr<IDialog_Meshgen> make_meshgen_dialog(QWidget *parent) {
    return std::unique_ptr<IDialog_Meshgen>();
}

#include "dlg_meshgen.moc"
