// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"

#include "dlg_meshgen.h"
#include "vm_meshgen.h"
#include <QDialog>
#include <marching_cubes.h>
#include <psp/psp.h>

class Dialog_Meshgen : public QDialog, public IDialog_Meshgen {
    Q_OBJECT;
    Q_INTERFACES(IDialog_Meshgen);

public:
    Dialog_Meshgen(QWorld const *world, Entity_Handle entity, QWidget *parent)
        : QDialog(parent)
        , _vm(world, entity) {
    }

    ~Dialog_Meshgen() override = default;

    void onRender(gfx::Render_Queue *rq) override {
    }

private:
    VM_Meshgen _vm;
};

IDialog_Meshgen *make_meshgen_dialog(QWorld const *world, Entity_Handle entity, QWidget *parent) {
    auto ret = new Dialog_Meshgen(world, entity, parent);
    ret->show();
    return ret;
}

#include "dlg_meshgen.moc"
