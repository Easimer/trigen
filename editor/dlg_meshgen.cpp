// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"

#include "dlg_meshgen.h"
#include "vm_meshgen.h"

#include <QBoxLayout>
#include <QDialog>
#include <QFileDialog>
#include <QLineEdit>
#include <QPushButton>

#include <marching_cubes.h>
#include <psp/psp.h>

#include <ui_dlg_meshgen.h>

class QTextureWidget : public QWidget {
    Q_OBJECT;

public:
    QTextureWidget(QWidget *parent)
        : QWidget(parent)
        , _layout(QBoxLayout::Direction::LeftToRight, this)
        , _editPath(this)
        , _btnBrowse("Browse...", this) {
        _layout.addWidget(&_editPath);
        _layout.addWidget(&_btnBrowse);
        connect(&_btnBrowse, &QPushButton::clicked, [&]() {
            auto path = QFileDialog::getOpenFileName(this, "Load a texture", QString(), "Images (*.png *.jpg *.bmp);;All files (*.*)");
            if (path.isEmpty()) {
                return;
            }

            _editPath.setText(path);
            emit pathChanged(path);
        });
    }

signals:
    void pathChanged(QString const &path);

private:
    QBoxLayout _layout;
    QLineEdit _editPath;
    QPushButton _btnBrowse;
};

class Dialog_Meshgen : public QDialog, public IDialog_Meshgen {
    Q_OBJECT;
    Q_INTERFACES(IDialog_Meshgen);

public:
    Dialog_Meshgen(QWorld const *world, Entity_Handle entity, QWidget *parent)
        : QDialog(parent)
        , _vm(world, entity) {
        _ui.setupUi(this);

        _vm.foreachInputTexture([&](Texture_Kind kind, char const *name, Input_Texture &tex) {
            auto texWidget = new QTextureWidget(this);
            _ui.layoutTextures->addRow(name, texWidget);
            connect(texWidget, &QTextureWidget::pathChanged, [&](QString const &path) {
                pathToTextureChanged(kind, path);
            });
        });
    }

    ~Dialog_Meshgen() override = default;

    void onRender(gfx::Render_Queue *rq) override {
    }

protected slots:
    void pathToTextureChanged(Texture_Kind kind, QString const &path) {
        auto pathUtf8 = path.toUtf8();
        _vm.loadTextureFromPath(kind, pathUtf8.constData());
    }

private:
    VM_Meshgen _vm;

    Ui::Dialog_Meshgen _ui;
};

IDialog_Meshgen *make_meshgen_dialog(QWorld const *world, Entity_Handle entity, QWidget *parent) {
    auto ret = new Dialog_Meshgen(world, entity, parent);
    ret->show();
    return ret;
}

#include "dlg_meshgen.moc"
