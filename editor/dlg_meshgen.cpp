// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"

#include "dlg_meshgen.h"
#include "vm_meshgen.h"
#include "widget_texture.h"

#include <QBoxLayout>
#include <QDialog>
#include <QFileDialog>
#include <QMessageBox>
#include <QPushButton>
#include <QStatusBar>

#include <ui_dlg_meshgen.h>

class Dialog_Meshgen : public Base_Dialog_Meshgen, public IMeshgen_Statusbar {
    Q_OBJECT;

public:
    Dialog_Meshgen(QWorld const *world, Entity_Handle entity, QWidget *parent)
        : Base_Dialog_Meshgen(parent)
        , _vm(world, entity, this)
        , _ui{}
        , _statusBar(this) {
        setAttribute(Qt::WA_DeleteOnClose);
        _ui.setupUi(this);

        connect(&_vm, &VM_Meshgen::exportError, [&](QString const &msg) {
            QMessageBox::critical(this, tr("Exporting has failed"), msg);
        });

        connect(&_vm, &VM_Meshgen::showExportFileDialog, [&]() {
            auto path = QFileDialog::getSaveFileName(this, "Export...", QString(), "Autodesk FBX (*.fbx)");
            if (path.isEmpty()) {
                return;
            }

            _vm.onExportPathAvailable(path);
        });

        connect(_ui.chkRenderNormals, &QCheckBox::stateChanged, &_vm, &VM_Meshgen::renderNormalsOptionChanged);

        auto btnExport = new QPushButton(tr("&Export..."));
        btnExport->setDefault(true);

        connect(btnExport, &QPushButton::clicked, &_vm, &VM_Meshgen::onExportClicked);
        connect(&_vm, &VM_Meshgen::exported, this, &QDialog::accepted);

        _ui.buttonBox->addButton(btnExport, QDialogButtonBox::ActionRole);
        connect(_ui.buttonBox, &QDialogButtonBox::rejected, this, &QDialog::rejected);

        _vm.foreachInputTexture([&](Meshgen_Texture_Kind kind, char const *name, Input_Texture &tex) {
            auto texWidget = new QTextureWidget(this);
            _ui.layoutTextures->addRow(name, texWidget);
            connect(texWidget, &QTextureWidget::pathChanged, [&, kind](QString const &path) {
                pathToTextureChanged(kind, path);
            });
        });

        connect(_ui.sbMetaballRadius, qOverload<double>(&QDoubleSpinBox::valueChanged), &_vm, &VM_Meshgen::metaballRadiusChanged);
        connect(_ui.sbNumSubdivisions, qOverload<int>(&QSpinBox::valueChanged), &_vm, &VM_Meshgen::numberOfSubdivionsChanged);
        connect(_ui.sbResolution, qOverload<int>(&QSpinBox::valueChanged), &_vm, &VM_Meshgen::resolutionChanged);

        // HACKHACKHACK(danielm): trigger the valueChanged signal for fields with default values
        _vm.resolutionChanged(_ui.sbResolution->value());
        _vm.numberOfSubdivionsChanged(_ui.sbNumSubdivisions->value());
        _vm.metaballRadiusChanged(_ui.sbMetaballRadius->value());

        connect(_ui.actionInspectUV, &QAction::triggered, &_vm, &VM_Meshgen::inspectUV);

        _ui.topLayout->addWidget(&_statusBar);
    }

    ~Dialog_Meshgen() override = default;

    void
    onRender(gfx::Render_Queue *rq) override {
        _vm.onRender(rq);
    }

    void
    onRenderTransparent(gfx::Render_Queue *rq) override {
        _vm.onRenderTransparent(rq);
    }

    void
    setMessage(char const *message) override {
        _statusBar.showMessage(tr(message));
    }

    void
    setBusy(bool isBusy) override {
        // TODO(danielm): spinner
        if (!isBusy) {
            _statusBar.clearMessage();
        }
    }

protected slots:
    void pathToTextureChanged(Meshgen_Texture_Kind kind, QString const &path) {
        auto pathUtf8 = path.toUtf8();
        _vm.loadTextureFromPath(kind, pathUtf8.constData());
    }

private:
    VM_Meshgen _vm;
    QStatusBar _statusBar;

    Ui::Dialog_Meshgen _ui;
};

Base_Dialog_Meshgen *make_meshgen_dialog(QWorld const *world, Entity_Handle entity, QWidget *parent) {
    return new Dialog_Meshgen(world, entity, parent);
}

#include "dlg_meshgen.moc"
